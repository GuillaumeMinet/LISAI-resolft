## Queue/Run Linking Ambiguity (Future Improvement)

### Issue
Queue jobs are often created with `run_id = null`, and the worker later backfills it using `_infer_run_id_for_job` (matching by `run_name + dataset + model_subfolder`, with active/stale heuristics).
This can be ambiguous when multiple candidate runs match, and dynamic stale logic can change which candidate is considered “active”, causing wrong linking or no linking.

### Proposed Fix
Use deterministic linking instead of inference:
1. Create/assign `run_id` at queue submit time for `train`/`retrain`.
2. For `continue_training`, reuse the existing run’s `run_id` (never create a new one).
3. Pass that `run_id` from queue worker to the training process.
4. Let run metadata creation accept an optional `run_id` override and persist it.
5. Keep `_infer_run_id_for_job` only as a legacy fallback for old jobs with missing `run_id`.

### Be Careful About
1. **Continue mode safety**: do not generate a new `run_id`; must match origin run metadata.
2. **Validation**: reject invalid/mismatched `run_id` early (fail fast).
3. **Backward compatibility**: keep old queue files working; keep fallback inference during migration.
4. **Schema strictness**: avoid adding extra metadata fields in phase 1 to prevent mixed-version breakage.
5. **Behavior stability**: keep current continue/stale/crash decision policy unchanged unless explicitly redesigned.
6. **Testing**: add coverage for ambiguous candidates, worker restart reconciliation, and continue-mode run_id reuse.
