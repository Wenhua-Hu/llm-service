# Experiments

The required `run.sh`, `kv.py`, `prefix.py`, and `attention.py` are not
implemented yet. This directory intentionally contains no fabricated experiment
results. The required CLI contract is:

```bash
bash experiments/run.sh --case kv --out results/<run-id>/kv
bash experiments/run.sh --case prefix --out results/<run-id>/prefix
bash experiments/run.sh --case attention --out results/<run-id>/attention
```
