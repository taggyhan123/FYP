# Accuracy at Qwen3-4B — stated before the runs

## Why

The accuracy cost of reordering is the one finding in this report most likely to
be an artefact of model size. It is measured only at Qwen3-0.6B, where absolute
accuracy is 11–28%, and a 0.6B model is plausibly unusually sensitive to how deep
in the menu the correct tool sits.

That sensitivity is load-bearing. It is the sole reason §5 rejects the canonical
and hybrid orderings: at k128 the hybrid buys +1.12pp reuse for −9.3pp accuracy,
an exchange rate of 8.3 accuracy points per point of reuse. If a larger model
tolerates depth better, that verdict changes.

Reuse and latency do NOT need re-running: reuse is already verified
model-invariant (1.19 / 37.99 / 87.21 / 96.16% at 4B against
1.19 / 38.13 / 87.19 / 96.16% at 0.6B).

## Predictions

1. Absolute accuracy rises on every arm at 4B.
2. Direction holds: reordering still costs accuracy against `original`.
   Lost-in-the-middle is a general phenomenon, not a 0.6B defect.
3. Magnitude shrinks: each arm's penalty is smaller at 4B than at 0.6B.
4. **Decisive.** `hybrid_oracle` at k128 costs 9.3pp at 0.6B. If that falls below
   ~2pp the exchange rate stops being prohibitive and §5's rejection of the
   canonical/hybrid line no longer holds.
5. **Validity.** `ceiling` — the share of requests whose menu contains a gold
   tool — is a property of the workload, not the model, so it MUST match 0.6B
   exactly: 0.755 at k64, 0.805 at k128. If it does not, the run is invalid.

## 0.6B baseline being compared against (gold_hit_ceil)

| depth | original | ContextPilot | tooltrie_v0 |
|---|---|---|---|
| k64 | 37.09% | 27.15% | 25.83% |
| k128 | 22.98% | 22.98% | 16.15% |

## Protocol

Identical to `accuracy-gate-20260830-011416`: serial `replay_vllm_workload.py`,
`--max-tokens 128 --tool-choice auto --disable-thinking --reset-before`, same
frozen workloads. Only the model and port change. k128 runs first because §5's
verdict rests on it.
