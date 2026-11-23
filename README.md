# Engram prefill cache for mlx-serve

Keep frequently requested Engram rows in RAM to reduce SSD reads during prompt prefill. The cache builder copies native FP8 row bytes and E8M0 scales from your local DeepSeek-V4.1-Flash-1017 weights. Rows outside the fixed cache remain available through the runtime’s existing LRU and SSD-backed table.

This repository contains a cache builder, a row-frequency profile manifest, and a runtime patch for the [two-Mac mlx-serve fork](https://github.com/daflood/mlx-serve/tree/b696e13dc6f07d9c0231686f72f267a752fc57d9). The matching model layout is [DeepSeek-V4.1-Flash-1017-MLX-Serve-Lossless](https://huggingface.co/ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-Lossless).

## Build a cache from your own weights

Download `engram-row-frequency-v1.zst` from this repository’s [v0.1.0 release](https://github.com/lgrimani/mlx-serve-engram-cache/releases/tag/v0.1.0) into the checkout. Use Python 3.10 or newer with the `zstd` command on your path. Run these commands from the repository root in your Python environment.

```sh
python3 -m pip install -r requirements.txt

# Preview the row allocation without reading weights or creating files.
python3 build_cache.py --profile engram-row-frequency-v1.zst --budget-gib 48 --plan

# Use a model directory with config.json, tokenizer.json and the native shards at its root or under engram/.
python3 build_cache.py --profile engram-row-frequency-v1.zst --budget-gib 48 \
  --model /path/to/model --output ./engram-pack
```

Choose a smaller `--budget-gib` to make a smaller pack from the same profile. The budget includes the fixed rows, index and loader reserve on **each** Mac. The output directory must be new, and its parent must have enough disk space for the pack. The builder verifies the profile checksum, checks the local table layout and rereads the finished files to verify their hashes.

## Enable the cache

Apply the patch to the tested fork revision, then follow that checkout’s [server build instructions](https://github.com/daflood/mlx-serve/blob/b696e13dc6f07d9c0231686f72f267a752fc57d9/docs/building.md#server-only).

```sh
patch_file="$PWD/patches/mlx-serve-engram.patch"
git clone --branch dsv41-two-macs https://github.com/daflood/mlx-serve.git
cd mlx-serve
git checkout b696e13dc6f07d9c0231686f72f267a752fc57d9
git apply --check "$patch_file"
git apply "$patch_file"
```

Give each rank the same pack contents. In the environment that launches each patched server or worker, set the absolute path to its local copy.

```sh
export MLX_SERVE_DSV41_ENGRAM_CACHE=/absolute/path/to/engram-pack
export MLX_SERVE_DSV41_ENGRAM_LRU_MB=8192
```

The second setting is optional. It gives the separate runtime LRU 8 GiB per rank to retain recently used rows. Its default is 256 MiB. This patch supports up to 48 GiB for the fixed cache and 56 GiB for both caches combined per rank. Allow additional memory for the model, context and operating system.

## What the profile measures

The 131.76 MiB download stores aggregate row IDs and frequencies from 31,119,433 input tokens across 16 intelligence benchmarks and six performance prompts. The benchmark inputs were tokenized and mapped to Engram rows offline. Each token contributes one logical reference in each of 48 buckets. These counts characterize prefill and include repeated prompt text. Decode traffic, SSD reads and throughput require live measurements. The suite breakdown and hashes are in [the manifest](profile-v1.json).

The builder ranks rows by frequency across the two layers, reserving at least one row per layer. At 48 GiB, all 174,974,556 observed rows fit. The builder distributes the remaining slots across the buckets in proportion to their unused capacity. Each packed row uses 256 bytes of native codes and eight bytes of scales. Model weights stay on the machine that builds the pack.

The patch includes portions of mlx-serve under its [MIT license](patches/LICENSE.mlx-serve).
