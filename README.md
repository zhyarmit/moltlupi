# MoltLUPI: inference pipeline

The pipeline follows the paper's three phases:

1. Score agent pairs, discover communities with weighted Leiden, complete internal pair scores, and filter by density.
2. Extract exact entity mentions with Gemma and annotate post behavior and entity roles with JEV.
3. Aggregate accepted evidence by entity and observation window into propagation, shared-objective, or insufficient-evidence assessments.

Training, teacher models, distillation losses, training labels, and pairwise classifier training code are not included. Python source files contain no comments or docstrings. Explanations are kept in this README.

## Release status

This package includes a working runner, the relevant inference functions extracted from the original annotation code, an offline synthetic example, and tests. It does **not** include the trained MoltLUPI checkpoint, frozen encoders, research embeddings, or original dataset. The ONNX interface is a new public inference boundary; it has been tested with a synthetic model, not with the paper's unavailable checkpoint.

The bundled example runs immediately. Running the actual detector requires an exported student model and compatible agent/window representations, or previously computed pair scores. Live annotation additionally requires a reachable Gemma service and a JEV API key. The example annotations are hand-constructed fixtures, not model outputs or paper results.

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For ONNX model inference:

```bash
python -m pip install -e '.[onnx]'
```

## Run the offline example

From the repository root:

```bash
python -m moltlupi.cli run --config examples/config.json
```

This uses synthetic saved scores and annotations and makes no network requests. It should produce three four-agent clusters: one propagation, one shared-objective, and one insufficient-evidence cluster. Results are written to `runs/example/`. A run refuses to overwrite an existing output directory; change `output_dir` for a new run.

## Run with the trained model

1. Place the exported student in `models/student.onnx`.
2. Supply `data/agent_windows.npz`, `data/windows.json`, and `data/posts.jsonl` using the formats below.
3. Edit `examples/config.live.json`. Set `artifact_model` to the exact installed Gemma tag corresponding to the intended model; the placeholder is deliberately not a guessed service tag.
4. Set `TYPESAFE_API_KEY` in your environment. Keep credentials out of the configuration and repository.
5. Start your Gemma service and run:

```bash
python -m moltlupi.cli run --config examples/config.live.json
```

All paths in a configuration are resolved relative to the configuration file. Live JEV requests send supplied post text to the JEV service. Gemma requests use the configured endpoint, which defaults to localhost. Live services may incur costs; the pipeline does not provision them or download models.

### Model boundary

The ONNX model must accept exactly one `float32` tensor of shape `[batch, 2D]` and return one tensor of shape `[batch]` or `[batch, 1]`. For agent representations `z_a` and `z_b`, the runner constructs:

```text
concat(abs(z_a - z_b), z_a * z_b)
```

Set `output_type` to `logits` if the export returns pre-sigmoid logits, or `probabilities` if it already returns sigmoid probabilities. Do not apply sigmoid twice. Export the trained model in evaluation mode with a dynamic batch dimension. Export/conversion must use the original architecture privately; an arbitrary `.pt` state dictionary cannot be loaded without its matching architecture. No training module is imported by this package.

The supplied representations must match the checkpoint's encoders, modality order, masking, normalization, window construction, and pooling. This runner does not recreate those representations from raw posts. Privileged wallet/token features must not be substituted for observable student inputs. ONNX omits the training implementation but does not make model weights or architecture secret.

### Agent/window representations

`agent_windows.npz` is a NumPy archive with three arrays:

| Array | Shape | Type |
| --- | --- | --- |
| `agent_id` | `[N]` | Unicode strings |
| `window_id` | `[N]` | Unicode strings |
| `joint_repr` | `[N, D]` | Finite numeric values |

Each row is one eligible agent/window representation. `(agent_id, window_id)` must be unique. No pickle-backed object arrays are accepted. Two agents are compared only in windows represented for both agents. The roster is inferred from these rows.

The release uses explicit, shared window IDs. It does not invent pair-specific window alignment, enforce the paper's minimum-post eligibility policy, select the latest windows, or perform feature encoding. Prepare the archive using the original preprocessing policy. Use saved scores if your existing cache cannot be losslessly converted to this representation.

### Observation windows

`windows.json` is an array:

```json
[
  {
    "window_id": "w0",
    "start": "2026-01-01T00:00:00Z",
    "end": "2026-01-02T00:00:00Z"
  }
]
```

Timestamps must include timezones. Intervals use `[start,end)`: the start is included and the end excluded. Different window IDs may overlap; scores and evidence remain associated with their declared windows.

### Posts

`posts.jsonl` contains one complete source post per line:

```json
{"record_id":"post_1","author":"agent_1","timestamp":"2026-01-01T12:00:00Z","text":"The complete post text."}
```

Supply all collected member posts in the intended observation windows. The runner selects every supplied in-window member post without sampling. It cannot detect posts absent from your input. Duplicate identical record IDs are deduplicated; conflicting duplicates are rejected. Blank text, missing timestamps, and malformed IDs are rejected.

Packets contain one retained cluster in one window shared by all members. A window with a member missing from the supplied posts is excluded and recorded in `packet_exclusions.json`; members are never silently dropped. Clusters with no eligible packet remain in the final output with insufficient evidence and `coverage_status: no_eligible_packet`.

### Previously computed scores

To avoid model export, replace `embeddings` and `model` in `discovery` with `scores`, pointing to JSONL:

```json
{"agent_a":"agent_1","agent_b":"agent_2","window_id":"w0","probability":0.94}
```

Scores must be finite probabilities in `[0,1]`. Pairs are unordered; duplicate or reversed entries for the same pair/window and self-pairs are rejected. Distinct window scores are aggregated using the configured mean or maximum.

Without a model, all supplied pairs form the initial graph candidate set; `sample_pairs` is not applied. An agent's eligible windows are inferred from score records unless an embedding archive is also supplied. Include every eligible window and every internal pair needed for retained communities. Missing eligible scores during completion raise an error; they are never replaced by zero. Pairs with no shared window make a community ineligible for density evaluation.

You can provide `scores`, `embeddings`, and `model` together to reuse existing scores and compute missing ones. All must use the same model and preprocessing policy. Input hashes are recorded for traceability, but they cannot establish that externally supplied scores were computed by the supplied model.

For large populations, sampling bounds the initial number of pairs. The current runner keeps sampled pairs and cached scores in memory; it is not a disk-backed population-scale scheduler. Supply an externally prepared score file or partition the workflow if memory is limiting.

## Separate phases and resumption

The phases also run independently:

```bash
python -m moltlupi.cli phase1 --config examples/config.live.json
python -m moltlupi.cli packets --config examples/config.live.json
python -m moltlupi.cli phase2 --config examples/config.live.json
python -m moltlupi.cli phase3 --config examples/config.live.json
```

Each phase output directory must be new. Annotation requests and responses are cached in `annotations.sqlite` in the run directory. To resume an interrupted phase 2, retain the database and configuration and choose a fresh sibling output directory:

```bash
python -m moltlupi.cli phase2 --config examples/config.live.json --output-dir runs/live/phase2_retry
python -m moltlupi.cli phase3 --config examples/config.live.json --annotations runs/live/phase2_retry/post_annotations.jsonl --output-dir runs/live/phase3_retry
```

Use the same run-directory parent for phase-2 retries to reuse the cache. Changing annotation settings requires a new run directory. Completed answers are reused, not resampled. Previously failed entity extractions remain recorded and are not automatically retried. Cached responses preserve failures and source-coverage limitations for inspection.

You may also supply an original `post_annotations.jsonl` export through `annotation.annotations`. Source hashes must match the packet's exact author, timestamp string, and text. Missing annotations remain excluded evidence and are counted; they do not trigger live requests in replay mode.

## Defaults and assessment semantics

| Setting | Default |
| --- | --- |
| Initial edge threshold | 0.85 |
| Density edge threshold | 0.85 |
| Minimum density | 0.80 |
| Minimum cluster size | 4 |
| Leiden partition | Weighted CPM |
| Resolution | 0.1 |
| Leiden seed / passes | 0 / 10 |
| Pair-window aggregation | Mean |
| Initial model-scored pair sample | 10,000,000, capped by roster pair count |
| Accepted annotation probability and confidence | Both at least 0.6 |
| Distinct supporting authors | At least 2 per entity/mechanism |

The initial partition is fixed before internal score completion. Completion does not merge clusters or rerun Leiden. Density uses all unordered internal pairs, including low scores. Cluster IDs are deterministic output labels for a given partition, not the research IDs used in the paper.

Phase 2 retains the original 28 dimensions, exact-substring validation, contextual entity roles, source hashing, truncation markers, uncertainty labels, and API response validation. It uses the original Gemma extraction settings and JEV question format. Moltbook as an exact entity name is ignored. API/model availability must be checked in the deployment environment; live services were not exercised when preparing this release.

Phase 3 first checks member authorship, source identity, successful extraction, operative context, non-criticism, and an accepted beneficiary/promoted-entity/action-destination role. It counts distinct authors separately for each normalized entity and mechanism. Repeated posts by one author do not meet the two-author threshold.

Categories have precedence `propagation > shared_objective > insufficient_evidence`. A propagation pattern requires at least two eligible authors sharing both a target and an influence mechanism. Potentially improper mechanisms are reported separately, with the original single-target restriction. A candidate flag is not a finding of collusion or harm.

### Shared-objective policy

The source implementation and the paper's appendix table use shared-target evidence for the shared-objective category, while the main text additionally describes compatible intended outcomes. The release exposes this distinction rather than silently changing the published workflow:

- `legacy_target` (default): retain the original shared-target criterion and display it under the paper's `shared_objective` label. The evidence field `shared_objective` separately records whether compatible objective labels were supported.
- `compatible_objective`: require at least two eligible authors sharing the same normalized entity and the same accepted objective category. Equality of rubric categories is a conservative operational proxy; it does not prove that two agents seek the same concrete outcome.

Changing the policy may change cluster counts. Propagation does not acquire an additional objective gate under either policy. Original four-way conclusions are retained as `legacy_conclusion`; `collusion_candidate` is not a fourth public category, and improper-conduct flags remain separate. Entity-level legacy conclusions and their original rationales are retained for audit.

All supporting posts and per-mechanism/per-objective author counts and fractions are retained. Cluster categories take the highest category among their window-specific assessments; authors and mechanisms are not pooled across windows to create new qualifying patterns. These outputs characterize evidence-supported subgroups, not every cluster member.

## Outputs

| File | Contents |
| --- | --- |
| `manifest.json` | Resolved configuration, input SHA-256 hashes, run status |
| `phase1/clusters.json` | Membership, density, mean score, shared windows |
| `phase1/pair_scores.jsonl` | Supplied and newly computed window-level scores |
| `phase1/rejected.json` | Communities rejected after size selection |
| `phase1/report.json` | Discovery counts and settings |
| `packets.jsonl` | Complete supplied in-window posts for annotation |
| `packet_exclusions.json` | Missing common windows or member posts |
| `phase2/post_annotations.jsonl` | Post dimensions, entity roles, coverage and source |
| `phase2/run_report.json` | Annotation/replay coverage and request counts |
| `phase3/case_assessments.jsonl` | Window-specific rules and supporting evidence |
| `phase3/cluster_assessments.jsonl` | Cluster-level categories and case references |
| `phase3/report.json` | Category counts, policy, thresholds, input hashes |

Live annotation also writes an advisory `potential_noise_agents.json` and extraction-failure details. This advisory does not remove members. Inspect evidence and exclusions before interpreting classifications.

## Validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests cover the offline three-phase run, ONNX pair features and sigmoid handling, score-cache validation, sampling, all 28 annotation dimensions with mocked service responses, cache reuse, confidence gates, objective policies, source integrity, window boundaries, and absence of source comments/docstrings.

These tests establish runner behavior on fixtures. They do not validate detector accuracy, reproduce the paper's population counts, test actual service credentials, or establish equivalence between an exported model and the original checkpoint.

## Preparing the GitHub release

Publish the package source, examples, tests, `pyproject.toml`, and this README. Add the actual inference model or a stable download location with its checksum, the compatible encoder/preprocessing specification or prepared embeddings, the final paper citation, and your chosen license before advertising the repository as a reproducible model release. No license is imposed by this generated package.

Keep original datasets, research annotations, API keys, caches, and run outputs outside the public commit unless you deliberately choose to release them. The included examples are synthetic. This package has not been pushed to GitHub.
