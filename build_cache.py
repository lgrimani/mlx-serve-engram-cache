#!/usr/bin/env python3
"""Build an mlx-serve v2 fixed Engram pack from native local tensors and a row-frequency profile."""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np


PAGE = 16384
LOAD_RESERVE = 9 << 20
FIXED_LIMIT = 48 << 30
ROW_BYTES = 264
BLOCK_ROWS = 65536


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align(n, step):
    return (n + step - 1) // step * step


def validate_profile(manifest, asset):
    require(manifest["format_version"] == 1, "Unsupported profile format")
    identity = manifest["asset"]
    require(asset.stat().st_size == identity["bytes"] and sha256_file(asset) == identity["sha256"],
            "Profile size or SHA-256 differs from its manifest")
    tables = manifest["checkpoint"]["tables"]
    buckets = manifest["buckets"]
    require(len(tables) > 0 and len({t["layer"] for t in tables}) == len(tables), "Invalid layer table")
    ordered = sorted(buckets, key=lambda b: (b["layer"], b["row_start"]))
    require(buckets == ordered, "Buckets must be ordered by layer and row")
    row_offsets = {table["layer"]: 0 for table in tables}
    offset = 0
    for bucket in buckets:
        layer = bucket["layer"]
        rows = bucket["row_count"]
        used = bucket["observed_rows"]
        require(layer in row_offsets and row_offsets[layer] == bucket["row_start"], "Invalid bucket row range")
        require(0 < rows <= 2**32 - 1 and 0 < used <= rows, "Invalid bucket size")
        bitmap_bytes = (rows + 7) // 8
        require(bitmap_bytes + used <= bucket["raw_bytes"] <= bitmap_bytes + 5 * used, "Invalid decoded frame size")
        require(bucket["offset"] == offset and bucket["length"] > 0, "Invalid frame range")
        offset += bucket["length"]
        row_offsets[layer] += rows
    require(offset == identity["bytes"], "Unaccounted profile bytes")
    for table in tables:
        require(row_offsets[table["layer"]] == table["rows"] and 0 < table["rows"] < 2**32, "Incomplete layer")
        require((table["head_dim"], table["wlen"], table["slen"]) == (256, 256, 8), "Unsupported native format")
    require(sum(b["observed_rows"] for b in buckets) == manifest["observed_rows"], "Profile row totals differ")
    require(sum(b["logical_references"] for b in buckets) == manifest["logical_references"],
            "Profile reference totals differ")


def bucket_counts(source, bucket):
    source.seek(bucket["offset"])
    compressed = source.read(bucket["length"])
    require(hashlib.sha256(compressed).hexdigest() == bucket["sha256"], "Profile frame digest mismatch")
    raw = subprocess.run(["zstd", "-q", "-d", "-c"], input=compressed, stdout=subprocess.PIPE, check=True).stdout
    require(len(raw) == bucket["raw_bytes"], "Incorrect decoded frame size")
    nrows = bucket["row_count"]
    nbytes = (nrows + 7) // 8
    flags = np.unpackbits(np.frombuffer(raw, dtype="u1", count=nbytes), bitorder="little")
    require(not flags[nrows:].any(), "Nonzero trailing bitmap bits")
    ids = np.flatnonzero(flags[:nrows]).astype(np.uint32)
    data = np.frombuffer(raw, dtype="u1", offset=nbytes)
    ends = np.flatnonzero(data < 128)
    require(len(ids) == len(ends) == bucket["observed_rows"] and ends[-1] + 1 == len(data),
            "Invalid count stream")
    starts = np.r_[0, ends[:-1] + 1]
    widths = ends - starts + 1
    require(np.all(widths <= 5) and np.all(data[ends] > 0) and np.all(data[ends[widths == 5]] < 16),
            "Invalid uint32 frequency")
    counts = (data[starts] & 127).astype(np.uint32)
    for index in range(1, int(widths.max())):
        active = widths > index
        counts[active] |= (data[starts[active] + index] & 127).astype(np.uint32) << (7 * index)
    require(int(counts.sum(dtype=np.uint64)) == bucket["logical_references"], "Frequency sum mismatch")
    return ids, counts


def fallback_quotas(available, needed):
    """Apportion zero-frequency slots among all heads by their remaining capacity."""
    if not needed:
        return [0] * len(available)
    total = sum(available)
    quotas = [needed * n // total for n in available]
    order = sorted(range(len(available)), key=lambda i: -(needed * available[i] % total))
    for i in order[:needed - sum(quotas)]:
        quotas[i] += 1
    return quotas


def select(manifest, asset, budget):
    validate_profile(manifest, asset)
    require(0 < budget <= FIXED_LIMIT, "The patched runtime supports fixed-cache budgets up to 48 GiB")
    tables, buckets = manifest["checkpoint"]["tables"], manifest["buckets"]
    indexes = sum(align(align(t["rows"], 64) // 8, PAGE) + align((t["rows"] + 511) // 512 * 4, PAGE) for t in tables)
    # Leave room for each layer's payload to be rounded up independently.
    slots = min(sum(t["rows"] for t in tables),
                (budget - indexes - LOAD_RESERVE - len(tables) * (PAGE - 1)) // ROW_BYTES)
    require(slots >= len(tables), "Budget is too small for the index, loader scratch, and at least one row per layer")
    histogram, best = Counter(), {}
    with asset.open("rb") as source:
        for b in buckets:
            ids, counts = bucket_counts(source, b)
            freq, multiplicity = np.unique(counts, return_counts=True)
            histogram.update(dict(zip(freq.tolist(), multiplicity.tolist())))
            i = int(counts.argmax())
            # The loader requires one resident row per table; reserve its most frequent row.
            key = (int(counts[i]), -int(b["row_start"] + int(ids[i])))
            best[b["layer"]] = max(best.get(b["layer"], (0, 0)), key)
    for count, _ in best.values():
        histogram[count] -= 1
    needed = min(slots, manifest["observed_rows"]) - len(best)
    threshold, ties = 0, 0
    for count in sorted(histogram, reverse=True):
        if needed <= histogram[count]:
            threshold, ties = count, needed
            break
        needed -= histogram[count]
    quotas = fallback_quotas([b["row_count"] - b["observed_rows"] for b in buckets],
                             max(0, slots - manifest["observed_rows"]))
    bits = {t["layer"]: np.zeros(align(t["rows"], 64) // 8, dtype=np.uint8) for t in tables}
    selected = Counter()
    with asset.open("rb") as source:
        for b, quota in zip(buckets, quotas):
            ids, counts = bucket_counts(source, b)
            base, layer = b["row_start"], b["layer"]
            reserved = ids.astype(np.int64) == (-best[layer][1] - base)
            keep = reserved | (counts > threshold)
            equal = np.flatnonzero((counts == threshold) & ~reserved)[:ties]
            keep[equal] = True
            ties -= len(equal)
            mask = np.zeros(b["row_count"], dtype=bool)
            mask[ids[keep]] = True
            if quota:
                mask[np.flatnonzero(~mask)[:quota]] = True
            chosen = np.flatnonzero(mask).astype(np.uint32) + np.uint32(base)
            selected[layer] += len(chosen)
            np.bitwise_or.at(bits[layer], chosen >> 3, (1 << (chosen & 7)).astype(np.uint8))
    bill = LOAD_RESERVE + indexes + sum(align(n * ROW_BYTES, PAGE) for n in selected.values())
    require(sum(selected.values()) == slots and bill <= budget and ties == 0, "Inconsistent row allocation")
    return bits, dict(selected), bill


def local_tables(model, checkpoint):
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    require(config.get("model_type") == checkpoint["model_type"] and
            all(text.get(key) == value for key, value in checkpoint["text_config"].items()),
            "Checkpoint Engram configuration differs from the profile")
    require(sha256_file(model / "tokenizer.json") == checkpoint["tokenizer_sha256"],
            "Checkpoint tokenizer differs from the profile")
    raw_map, json_map = model / "engram-token-map.u32", model / "engram_token_map.json"
    if raw_map.exists() or json_map.exists():
        if raw_map.exists():
            digest = sha256_file(raw_map)
        else:
            values = np.array(json.loads(json_map.read_text()))
            require(values.dtype.kind in "ui" and values.ndim == 1 and len(values) == text["vocab_size"] and
                    int(values.min()) >= 0 and int(values.max()) < text["engram_compressed_vocab_size"],
                    "Invalid Engram token map")
            digest = hashlib.sha256(values.astype("<u4").tobytes()).hexdigest()
        require(digest == checkpoint["token_map_u32le_sha256"], "Engram token map differs from the profile")

    sources = {}
    for table in checkpoint["tables"]:
        name = table["source_file"]
        require(Path(name).name == name and name.endswith(".safetensors"), "Invalid source shard filename")
        path = model / "engram" / name
        if not path.exists():
            path = model / name
        require(path.stat().st_size == table["source_size"], f"Source shard size mismatch for {name}")
        with path.open("rb") as f:
            prefix = f.read(8)
            require(len(prefix) == 8, "Truncated safetensors length")
            size = struct.unpack("<Q", prefix)[0]
            require(2 <= size <= 16 << 20, "Invalid safetensors header length")
            header = f.read(size)
        require(len(header) == size and hashlib.sha256(header).hexdigest() == table["source_header_sha256"],
                f"Source shard header mismatch for {name}")
        tensors = json.loads(header)
        for kind, stride, prefix_key, dtype in (("weight_tensor", 256, "w", "F8_E4M3"),
                                                ("scale_tensor", 8, "s", "F8_E8M0")):
            tensor = tensors[table[kind]]
            a, z = tensor["data_offsets"]
            require(tensor["dtype"] == dtype and tensor["shape"] == [table["rows"], stride] and
                    table[prefix_key + "_stride"] == stride and size + 8 + a == table[prefix_key + "_offset"] and
                    z - a == table["rows"] * stride and 0 <= a <= z <= table["source_size"] - size - 8,
                    f"Native source tensor layout mismatch for {name}")
        sources[table["layer"]] = path
    return sources


def native_chunks(source, table, bits):
    for stride, offset in ((256, table["w_offset"]), (8, table["s_offset"])):
        for start in range(0, table["rows"], BLOCK_ROWS):
            n = min(BLOCK_ROWS, table["rows"] - start)
            packed = bits[start // 8 : (start + n + 7) // 8]
            if not packed.any():
                continue
            mask = np.unpackbits(packed, bitorder="little", count=n).astype(bool)
            if np.count_nonzero(mask) <= 8:
                # Sparse budgets can touch distant rows without pulling intervening pages into RAM.
                for row in np.flatnonzero(mask):
                    source.seek(offset + (start + int(row)) * stride)
                    raw = source.read(stride)
                    require(len(raw) == stride, "Truncated native row")
                    yield raw
                continue
            source.seek(offset + start * stride)
            raw = source.read(n * stride)
            require(len(raw) == n * stride, "Truncated native tensor")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(n, stride)[mask].tobytes()


def build(output, tables, sources, bits, selected):
    """An exclusive directory and a manifest published last protect retries and concurrent builders."""
    output.mkdir(parents=False, exist_ok=False)
    manifest = {"version": 2, "total_payload_bytes": sum(selected.values()) * ROW_BYTES, "layers": []}
    try:
        for table in tables:
            layer = table["layer"]
            bitmap = output / f"L{layer}.bitmap"
            payload = output / f"L{layer}.bin"
            bits[layer].tofile(bitmap)
            sha = hashlib.sha256()
            with sources[layer].open("rb") as source, payload.open("xb") as dst:
                for raw in native_chunks(source, table, bits[layer]):
                    dst.write(raw)
                    sha.update(raw)
            size = payload.stat().st_size
            require(size == selected[layer] * ROW_BYTES, "Incorrect packed payload length")
            item = {key: table[key] for key in ("layer", "source_file", "source_size", "source_header_sha256",
                    "rows", "head_dim", "wlen", "slen", "w_offset", "s_offset", "w_stride", "s_stride")}
            item.update(payload_file=payload.name, payload_bytes=size, payload_sha256=sha.hexdigest(), buckets=[],
                        selected_rows=selected[layer], bitmap_file=bitmap.name, bitmap_bytes=bitmap.stat().st_size,
                        bitmap_sha256=hashlib.sha256(bits[layer]).hexdigest())
            require(sha256_file(bitmap) == item["bitmap_sha256"] and sha256_file(payload) == item["payload_sha256"],
                    "On-disk pack failed verification")
            manifest["layers"].append(item)
        partial = output / "manifest.json.partial"
        partial.write_text(json.dumps(manifest, indent=2) + "\n")
        partial.rename(output / "manifest.json")
    except BaseException:
        # Only this invocation could have created the exclusive directory.
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Downloaded compressed frequency profile")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("profile-v1.json"))
    parser.add_argument("--budget-gib", type=str, required=True, help="Per-rank fixed cache RAM budget (GiB)")
    parser.add_argument("--model", type=Path, help="Native checkpoint with config.json, tokenizer.json and FP8 shards at its root or in engram/")
    parser.add_argument("--output", type=Path, help="New pack directory; its parent must exist")
    parser.add_argument("--plan", action="store_true", help="Print row allocation without opening the model or writing a pack")
    args = parser.parse_args()
    if not args.plan and (args.model is None or args.output is None):
        parser.error("Provide --model and --output to build, or use --plan")
    try:
        budget = Decimal(args.budget_gib)
    except InvalidOperation:
        parser.error("--budget-gib must be a number")
    if not budget.is_finite() or not 0 < budget <= 48:
        parser.error("--budget-gib must be positive and at most 48")
    manifest = json.loads(args.manifest.read_text())
    if not args.plan:
        if os.path.lexists(args.output):
            parser.error("Output already exists")
        sources = local_tables(args.model, manifest["checkpoint"])
    bits, selected, bill = select(manifest, args.profile, int(budget * (1 << 30)))
    total = sum(selected.values())
    observed = min(total, manifest["observed_rows"])
    print(f"Selected {total:,} rows ({observed:,} observed, {total-observed:,} fallback).")
    print(f"Fixed allocation including index and reserve is {bill / 2**30:.4f} GiB per rank.")
    if not args.plan:
        build(args.output, manifest["checkpoint"]["tables"], sources, bits, selected)
        print(f"Verified cache pack at {args.output}")


if __name__ == "__main__":
    main()
