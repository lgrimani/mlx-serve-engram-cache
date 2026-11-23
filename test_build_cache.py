"""Check selection, v2 native packing, and integrity on an offline two-layer fixture."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import build_cache as cache


def digest(data):
    return hashlib.sha256(data).hexdigest()


def leb128(n):
    data = bytearray()
    while n >= 128:
        data.append((n & 127) | 128)
        n >>= 7
    data.append(n)
    return data


def fixture(root):
    model = root / "model"
    (model / "engram").mkdir(parents=True)
    text = {"engram_layer_ids": [1, 14], "engram_num_embeddings": [130, 130],
            "vocab_size": 3, "engram_compressed_vocab_size": 3}
    (model / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "text_config": text}))
    (model / "tokenizer.json").write_bytes(b"{}")
    checkpoint = {"model_type": "deepseek_v41", "text_config": text, "tables": [],
                  "tokenizer_sha256": digest(b"{}"), "token_map_u32le_sha256": digest(struct.pack("<3I", 0, 2, 1))}
    # Bucket two starts on an unaligned byte. Tied counts straddle heads and layers.
    freqs = [{0: 129, 1: 4, 63: 3, 64: 3}, {0: 16384, 64: 2},
             {0: 3, 1: 128, 64: 4}, {0: 2, 64: 9}]
    frames = bytearray()
    buckets = []
    for i, layer in enumerate(text["engram_layer_ids"]):
        weight = f"layers.{layer}.engram.embed.weight"
        scale = f"layers.{layer}.engram.embed.scale"
        wbytes = 130 * 256
        sbytes = 130 * 8
        header = json.dumps({weight: {"dtype": "F8_E4M3", "shape": [130, 256], "data_offsets": [0, wbytes]},
                             scale: {"dtype": "F8_E8M0", "shape": [130, 8], "data_offsets": [wbytes, wbytes+sbytes]}}).encode()
        header = header.ljust(cache.align(len(header), 8), b" ")
        w = b"".join(bytes((row + col + layer) % 256 for col in range(256)) for row in range(130))
        s = b"".join(bytes((3 * row + col + layer) % 256 for col in range(8)) for row in range(130))
        shard = model / "engram" / f"layer{layer}.safetensors"
        shard.write_bytes(struct.pack("<Q", len(header)) + header + w + s)
        offset = 8 + len(header)
        checkpoint["tables"].append(dict(layer=layer, source_file=shard.name, source_size=shard.stat().st_size,
            source_header_sha256=digest(header), rows=130, head_dim=256, wlen=256, slen=8, w_offset=offset,
            s_offset=offset + wbytes, w_stride=256, s_stride=8, weight_tensor=weight, scale_tensor=scale,
            weight_dtype="F8_E4M3", scale_dtype="F8_E8M0"))
        for head in range(2):
            counts = freqs[2*i + head]
            bits = bytearray(9)
            for index in counts:
                bits[index // 8] |= 1 << (index % 8)
            raw = bits + b"".join(leb128(n) for n in counts.values())
            frame = subprocess.run(["zstd", "-q", "-c"], input=raw, stdout=subprocess.PIPE, check=True).stdout
            buckets.append(dict(layer=layer, ngram_order=2, head=head, row_start=65*head, row_count=65,
                observed_rows=len(counts), logical_references=sum(counts.values()), offset=len(frames),
                length=len(frame), raw_bytes=len(raw), sha256=digest(frame)))
            frames.extend(frame)
    asset = root / "profile.zst"
    asset.write_bytes(frames)
    manifest = dict(format_version=1, checkpoint=checkpoint, buckets=buckets,
                    observed_rows=sum(len(c) for c in freqs), logical_references=sum(sum(c.values()) for c in freqs),
                    asset=dict(filename=asset.name, bytes=len(frames), sha256=digest(frames)))
    (root / "profile.json").write_text(json.dumps(manifest))
    return model, manifest, asset


def budget_for(rows):
    return (cache.LOAD_RESERVE + 4 * cache.PAGE + 2 * (cache.PAGE - 1) + rows * cache.ROW_BYTES)


def rows_in(bitmap):
    return np.flatnonzero(np.unpackbits(bitmap, bitorder="little")).tolist()


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.manifest, self.asset = fixture(self.root)

    def test_ranking_budget_and_ties(self):
        for k, expected in [(2, {1: [65], 14: [1]}), (5, {1: [0, 1, 65], 14: [1, 129]}),
                            (7, {1: [0, 1, 63, 65], 14: [1, 64, 129]})]:
            bits, selected, fixed = cache.select(self.manifest, self.asset, budget_for(k))
            self.assertEqual({layer: rows_in(b) for layer, b in bits.items()}, expected)
            self.assertEqual(sum(selected.values()), k)
            self.assertLessEqual(fixed, budget_for(k))
        for budget in [budget_for(1), 48 * (1 << 30) + 1]:
            with self.assertRaises(ValueError):
                cache.select(self.manifest, self.asset, budget)

    def test_fallback_fills_unseen_across_heads(self):
        bits, _, _ = cache.select(self.manifest, self.asset, budget_for(15))
        self.assertEqual(rows_in(bits[1]), [0, 1, 2, 63, 64, 65, 66, 129])
        self.assertEqual(rows_in(bits[14]), [0, 1, 2, 64, 65, 66, 129])

    def test_native_payload_and_existing_output(self):
        cp = self.manifest["checkpoint"]
        sources = cache.local_tables(self.model, cp)
        bits, selected, _ = cache.select(self.manifest, self.asset, budget_for(35))
        output = self.root / "pack"
        cache.build(output, cp["tables"], sources, bits, selected)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual((manifest["version"], manifest["total_payload_bytes"]), (2, 35 * cache.ROW_BYTES))
        for table, layer in zip(cp["tables"], manifest["layers"]):
            bitmap = (output / layer["bitmap_file"]).read_bytes()
            self.assertEqual(len(bitmap), 24)
            self.assertEqual(digest(bitmap), layer["bitmap_sha256"])
            rows = rows_in(np.frombuffer(bitmap, np.uint8))
            native = sources[layer["layer"]].read_bytes()
            expected = bytearray()
            for offset, stride in [(table["w_offset"], 256), (table["s_offset"], 8)]:
                for row in rows:
                    start = offset + row * stride
                    expected.extend(native[start:start+stride])
            self.assertEqual((output / layer["payload_file"]).read_bytes(), expected)
            self.assertEqual(digest(expected), layer["payload_sha256"])
            self.assertEqual(len(rows), layer["selected_rows"])
        with self.assertRaises(FileExistsError):
            cache.build(output, cp["tables"], sources, bits, selected)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        with patch.object(cache, "native_chunks", side_effect=ValueError("Truncated")):
            with self.assertRaisesRegex(ValueError, "Truncated"):
                cache.build(self.root / "interrupted", cp["tables"], sources, bits, selected)
        self.assertFalse((self.root / "interrupted").exists())

    def test_input_integrity(self):
        cp = self.manifest["checkpoint"]
        sources = cache.local_tables(self.model, cp)
        override = self.model / "engram_token_map.json"
        override.write_text("[0, 2, 1]")
        cache.local_tables(self.model, cp)
        override.write_text("[0, 1, 2]")
        with self.assertRaisesRegex(ValueError, "token map"):
            cache.local_tables(self.model, cp)
        override.unlink()
        with sources[1].open("r+b") as f:
            f.seek(8)
            f.write(b"!")
        with self.assertRaisesRegex(ValueError, "header"):
            cache.local_tables(self.model, cp)
        (self.model / "tokenizer.json").write_text("different")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            cache.local_tables(self.model, cp)
        self.manifest["buckets"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            cache.select(self.manifest, self.asset, budget_for(10))
        with self.asset.open("r+b") as f:
            f.write(b"!")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cache.validate_profile(self.manifest, self.asset)


if __name__ == "__main__":
    unittest.main()
