# Performance experiments

## Status

No SGLang profile has been launched or measured on the taskbook RTX 4090
environment. All performance results are therefore **pending**. This document
does not claim any throughput, latency, cache hit rate, or correctness result.

## Required profiles

`configs/baseline.yaml` is the conservative comparison profile. The planned
optimized profile enables prefix caching (SGLang RadixAttention) and uses the
FlashInfer attention backend. The KV token budgets must be measured on the GPU;
the optimized budget must be greater than the baseline budget. Both YAML files
currently mark this calibration as pending and leave `max_total_tokens` unset.

## Required experiments

The taskbook requires repeatable KV cache on/off, prefix cache on/off, and
math/FlashAttention comparisons with raw records, manifests, correctness checks,
and generated summaries. The `experiments/` entrypoint and those experiment
implementations have not been added yet. Once run, link the exact commit and
unmodified evidence files here; do not substitute the public practice benchmark
or CPU fixture results for GPU performance measurements.

## Frozen performance targets

The taskbook's accepted thresholds are recorded in
`docs/requirements/acceptance.json`. They are evaluation targets, not this
project's measurements. No pass/fail result is available until the full workload
is run against the final optimized service.
