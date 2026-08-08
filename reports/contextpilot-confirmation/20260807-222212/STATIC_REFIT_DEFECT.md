# Blocking defect: `static_refit_causal` cannot run against pinned upstream

Found during section 2 of `NUS_GPU_CONTEXTPILOT_CONFIRMATION_INSTRUCTIONS.md`,
before any GPU work. Recorded here so the failure is preserved rather than
worked around.

## Symptom

```
$ "$CONTEXTPILOT_VENV/bin/python" scripts/build_contextpilot_workload.py \
    --input .../toolret-bm25-k4-original.jsonl --mode static_refit_causal --alpha 0.001 ...
ValueError: ContextPilot changed the selected tool set for input record 0
```

Raised by `_validate_ordering` at `src/tatm/contextpilot_adapter.py:33`, called
from `build_static_refit_causal_orderings` at line 71.

## Cause

`ContextIndex.fit_transform` returns contexts in ContextPilot's **internal
integer-ID space**, not in the string space the caller passed in.

Direct reproduction against pinned upstream `1fa0a143`:

```
selected  : ['toolACE_tool_14425', 'toolbench_tool_8962',
             'toolACE_tool_15523', 'toolbench_tool_13036']
returned  : ['0', '1', '2', '3']
set equal : False
```

Upstream `contextpilot/context_index/index_construction.py`:

- line 148-149 docstring: *"each context is a list of chunk IDs (int) or
  strings. String inputs are automatically converted to integer IDs."*
- line 154: `contexts = self._convert_to_int(contexts)` populates `self._str_to_id`
  and `self._id_to_str`.
- line 134: `_convert_to_str` exists and would reverse this, but **it is never
  called from `fit_transform`**. Neither the `n < 2` branch
  (`_handle_single_prompt`, line 234) nor the main return (line 199) invokes it.

So `fit_transform` returns integer IDs for **every** record, not only the first.
The adapter's validation is correct; it is the missing inverse mapping that is
the defect.

## Scope

| Arm | API | Status |
| --- | --- | --- |
| `static_refit_causal` | `ContextIndex.fit_transform` | **BLOCKED** — every record fails validation |
| `online_incremental` | `contextpilot.server.live_index.ContextPilot.reorder` | **WORKS** — returns string IDs; 5-record probe built cleanly |

Verified: the persistent-API arm is unaffected. Only the static-refit arm is
blocked.

## Why the test suite did not catch this

`tests/test_contextpilot_adapter.py:70` defines a `FakeIndex` whose
`fit_transform` returns `[list(reversed(context)) for context in contexts]` —
it echoes back the string IDs it was handed. The real upstream object does not.

The 104 passing tests therefore never exercised
`build_static_refit_causal_orderings` against real upstream. "All 104 tests
pass" is not evidence that this arm works.

## Proposed fix (NOT applied — needs owner sign-off)

`IndexResult` carries `original_contexts`, which is the *post-conversion*
integer form of the input, positionally aligned with the string list the caller
passed. The reordering can therefore be mapped back using only public
`IndexResult` fields, with no reliance on the private `_id_to_str`:

```python
int_source = list(result.original_contexts[-1])   # ints, aligned with `selected`
position_of = {value: i for i, value in enumerate(int_source)}
ordered = [selected[position_of[value]] for value in result.reordered_contexts[-1]]
```

`_validate_ordering` then still guards membership, so a genuine set change would
still be caught.

This was left unapplied because the adapter is the local analysis session's
deliverable, and choosing how to read an upstream API is a methodological
decision that must be owned and documented by whoever declares the arm's label,
not silently patched inside the GPU executor session.

## Effect on this run

Everything except `static_refit_causal` was executed. Because
`replay_vllm_workload.py` resets the prefix cache before every trial, each
condition's trials are independent measurements — the missing arm can be added
later without invalidating the arms measured here.
