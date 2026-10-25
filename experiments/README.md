# GPU mechanism experiments

`run.sh` dispatches to the KV-cache, SGLang RadixAttention, and PyTorch SDPA
math/FlashAttention experiments. Dependencies are installed by `setup.sh`; the
runner does not install or upgrade packages. Every invocation requires a new
output directory and writes a manifest plus raw evidence. Numeric failures keep
their measurements and return nonzero; interrupted runs are marked incomplete.

The experiment code has not yet been run on the taskbook RTX 4090. No GPU
measurements or pass claims are included in this repository.

CLI contract:

```bash
bash experiments/run.sh --case kv --out results/<run-id>/kv
bash experiments/run.sh --case prefix --out results/<run-id>/prefix
bash experiments/run.sh --case attention --out results/<run-id>/attention
```

KV and attention runs have a 300-second limit. Prefix runs have a 480-second
limit. The prefix experiment restarts SGLang with RadixAttention enabled and
disabled, flushes the cache before every measured group, and records the exact
input IDs and server-reported cached-token counts.
