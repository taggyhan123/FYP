# Initial-brief GPU gap closure

The predeclared experiment is
[`cluster/initial-brief-closure-manifest.json`](../../cluster/initial-brief-closure-manifest.json),
and the executable server procedure is
[`NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md`](../../NUS_GPU_BRIEF_CLOSURE_INSTRUCTIONS.md).

This directory is reserved for compact, reviewable summaries produced from the
GPU run. Raw workloads, per-request results, token IDs, and server logs remain
under the ignored `cluster/results/` directory and in a checksummed archive on
the GPU server. No GPU result should be added here until the manifest's validity
conditions have been checked.

The GPU handover must include:

- the exact FYP commit and clean/dirty status at execution time;
- GPU, CUDA, model, tokenizer, vLLM, block-size, and server-flag provenance;
- one summary per menu size for all declared replay conditions;
- compact rendered-prefix audit and pressure summaries;
- the raw archive path and SHA-256 checksum;
- every failed, quarantined, or missing condition.

An incomplete run is useful evidence but must not be labelled as closure of the
initial brief.
