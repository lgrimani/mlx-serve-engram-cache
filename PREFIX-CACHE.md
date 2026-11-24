# Prefix caching for long conversations

A long conversation can trigger a full prefill even when only the final instruction changes. In one review/resume cycle, the client sent the same history first with a reviewer instruction and then with a continuation instruction. The reviewer checkpoint extended 313 tokens beyond the point where the prompts diverged. The multimodal restore path required a whole checkpoint match. A preceding checkpoint matched 669,972 of the continuation's 670,779 tokens, but the 8 GiB prefix cache had just evicted it. Keeping it would have left 807 tokens to process.

This optional patch targets the mlx-serve fork pinned below. The prefix cache holds model state for conversation prefixes. The Engram cache described in the README holds model lookup-table rows. Each patch can be applied independently.

## What changes

| Change | Behavior |
| --- | --- |
| Multimodal checkpoint eviction | Opt in with `MLX_SERVE_PREFIX_PREFER_PARENT=1`. When an extending prompt needs space and byte-budget eviction picks the whole prompt checkpoint it just restored, prefer an older eligible checkpoint with the same workload key and model binding. If room still runs out, the parent can be evicted. The preference is consumed on the extending commit. Requests sharing a key can belong to different conversations. |
| Scheduling text requests during a long prefill | Included when the patch is applied. The DeepSeek V4.1 scheduler estimates remaining work from resident module checkpoints and reuses its estimate between chunks while the cache activity counter and entry count stay the same. Admission checks the actual restored length. The ordinary limit is 1,024 uncached tokens. Requests with images still follow the ordinary queue. |
| Diagnosing evictions | Opt in with `MLX_SERVE_PREFIX_DIAG=1`. An in-memory ring retains token IDs and media-match metadata for the last 100 requests. On a cold miss, the diagnostic log records prefix lengths, rejection reasons and compatible evicted checkpoints. Restarting clears the ring. At 100 prompts of 650k tokens each, token IDs alone occupy about 248 MiB. |

In a live two-Mac test, a 4,114-token text request arrived during a 704,250-token prefill. It restored 4,104 tokens and the scheduler admitted the remaining 10 between chunks. The follow-up completed in about 0.4 seconds. This is one observed request. The eviction example above is a diagnosis from before the policy change.

## Apply and test

From this repository, save the patch path, then switch to a clean mlx-serve checkout at the tested revision. An existing checkout with the Engram patch applied also works.

```sh
prefix_patch="$PWD/patches/mlx-serve-prefix-cache.patch"
cd /path/to/mlx-serve
git checkout b696e13dc6f07d9c0231686f72f267a752fc57d9
git apply --check "$prefix_patch"
git apply "$prefix_patch"
zig build test -Dtest-filter=boundary
zig build test -Dtest-filter='module byte budget preserves just restored'
zig build test -Dtest-filter='one hundred prompts rotate'
zig build -Doptimize=ReleaseFast -Dunits=true
```

Use the fork's pinned Zig toolchain and [build dependencies](https://github.com/daflood/mlx-serve/blob/b696e13dc6f07d9c0231686f72f267a752fc57d9/docs/building.md#server-only). For a distributed deployment, install matching binaries on both ranks. Set the desired flags in the launch environment and restart. The reported run used `--prefix-cache-mem 8GB`.

## Check whether evictions hurt

With diagnostics enabled, run the following from this repository for one or more coordinator logs. The script reads local files and needs only the Python standard library.

```sh
python3 report_prefix_evictions.py /path/to/rank0.log
```

Compare the number of **cold requests with a confirmed evicted match** per total request under similar traffic. The report counts each affected request once, attributes its longest compatible evicted checkpoint to the recorded rule, and gives the log line to inspect. Repeated misses attributed to `same_workload_older_checkpoint` mean later requests needed entries the new preference displaced. The report covers cold misses for which comparison metadata survived in the diagnostic window. Partial hits and earlier history fall outside that measure.

For scheduling, look for a `[boundary] admitted` line and its preceding `[prefix-diag] request=` line. Subtract restored tokens from prompt tokens to get the actual uncached work. Coordinator logs can also contain ordinary prompt previews, so share aggregate counts or selected numeric lines when reporting results.

## Revert

Unset `MLX_SERVE_PREFIX_PREFER_PARENT` and restart to use the original eviction policy. Unset `MLX_SERVE_PREFIX_DIAG` to disable the diagnostic history on the next launch. To remove the scheduler change as well, restore the previous binary on each rank, or reverse this patch in the same source checkout and rebuild:

```sh
git apply --check -R "$prefix_patch"
git apply -R "$prefix_patch"
zig build -Doptimize=ReleaseFast -Dunits=true
```

Restarting clears resident conversation checkpoints, so the first use of a long prompt may need a cold prefill.
