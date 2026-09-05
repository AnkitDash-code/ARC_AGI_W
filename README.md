# Project Mythos — ARC-AGI-2 Solver

**Status: active again -- see "Agentic program synthesis pivot" below.** The neural
(HRM/TTT/LoRA) path documented in the rest of this README was shelved (record preserved
below for why). Since then, a new architecture was built and validated with real,
verified results on real hardware: an LLM-driven program-synthesis solver
(`agentic_repl/`) that already measurably beats this project's own documented
symbolic-solver baseline. Currently paused only because Kaggle's weekly GPU quota was
exhausted mid-benchmark (refreshes 2026-08-01), not because the approach stopped
working.

## Agentic program synthesis pivot (current work, real results)

`agentic_repl/` (a separate top-level package from the shelved `src/mythos` neural
path) has an LLM (`Qwen3-Coder-30B-A3B-Instruct`, GGUF Q4_K_M, run via
`llama-cpp-python`) generate candidate `solve(grid)` Python programs against a DSL
built from this project's own verified grid primitives (`mythos.objects`/`object_ops`/
`symmetry`/`augment`), verifies every candidate against every train pair in a sandboxed
subprocess REPL before ever trusting it, refines failing candidates against concrete
failure deltas, and majority-votes across D4-augmented views. See
`agentic_repl/models/README.md` for the full staging story (offline wheel/model
staging for Kaggle's internet-disabled L4 sessions) and the git history of
`scripts/build_standalone_notebook.py` for the real bugs found and fixed getting this
running on actual Kaggle L4 hardware (wrong LLM completion API for an instruct model,
a refinement prompt that silently dropped train-example context, a context-window
overflow).

**Real, verified results** (composed with the existing symbolic solver -- it never
guesses wrong on train, so running it first only adds coverage, never subtracts),
against the real public ARC-AGI-2 training split, non-overlapping task slices so no
result is inflated by re-testing the same tasks twice:

| Tasks (offset) | Exact matches | Rate | Symbolic alone | agentic_repl's unique contribution |
| --- | --- | --- | --- | --- |
| 0–100  | 5/100  | 5.0% | 3 | 2 new solves (`08ed6ac7`, `1d0a4b61`) |
| 100–300 | 13/200 | 6.5% | 5 | 8 new solves |
| **Cumulative 0–300** | **18/300** | **6.0%** | 8 | **10 new solves beyond symbolic** |

This project's own documented symbolic-solver baseline (see "What actually worked"
below) is **25/1000 (2.5%)** on the full training split. 6.0% on a 300-task slice is
more than double that -- a real, measured improvement over the current best result in
this repository, not just a comparable one. Next step (once GPU quota refreshes):
continue covering the remaining training tasks (`MYTHOS_TASK_OFFSET`/`MYTHOS_MAX_TASKS`
in the generated notebook) for a full-1000 comparable number, and the held-out
evaluation split (120 tasks).

## Held-out evaluation split (120 tasks)

**Only the symbolic-solver half of this comparison could be reproduced in the local
dev environment this section was written in.** The environment has no GPU, no
`llama-cpp-python` installed, and no staged `Qwen3-Coder-30B-A3B-Instruct` GGUF (the
30B model that actually produced the 18/300 training-split result above was run on a
Kaggle L4 session, per "Agentic program synthesis pivot" above) -- running it locally
would mean either a multi-hour+ CPU inference per task or fabricating a number, and
per this project's own measurement rule ("every accuracy claim must be measured with
the real benchmark harness against a real task slice, never assumed"), neither is
acceptable. What *was* verified locally, against the real public ARC-AGI-2 dataset
(cloned fresh from `arcprize/ARC-AGI-2`, not the toy fixtures):

| Slice | Solver | Exact tasks | Rate |
| --- | --- | --- | --- |
| Training (0-1000) | Symbolic only | 25/1000 | 2.5% | (reproduces the documented number above exactly -- confirms the harness and freshly-cloned dataset are faithful)
| **Evaluation (120, held-out)** | **Symbolic only** | **1/120** | **0.8%** |

Reproduce with:
```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git <src>
python scripts/convert_arc_agi_2_data.py --src <src> --split evaluation --out-dir <out>
python scripts/benchmark_symbolic.py --challenges <out>/evaluation_challenges.json --solutions <out>/evaluation_solutions.json
```

**Decision gate: unresolved, not passed.** The relative symbolic-only drop
(2.5%->0.8%) is large, but the sample is only 1 task and symbolic-only was never the
subject of this gate -- the gate is about whether `agentic_repl`'s *combined* result
generalizes, and that number cannot be produced without the real LLM. Per this
project's rule against assuming accuracy from code review, this section stops here
rather than guessing whether the gate would pass or fail. **To actually resolve it:**
run
```bash
MYTHOS_AGENTIC_MODEL_PATH=<staged .gguf path> python scripts/benchmark_agentic_repl.py \
    --challenges <out>/evaluation_challenges.json --solutions <out>/evaluation_solutions.json \
    --llm-client llamacpp
```
on a Kaggle session with the model staged (same setup as the training-split run
above), the next time GPU quota is available.

Because the gate is unresolved rather than confirmed-passing, Steps 1-2 below (pure
control-flow / prompt-quality changes, independent of whether `agentic_repl` itself
generalizes) were implemented and measured with what's available locally. Step 3 (the
solved-program retrieval library) was intentionally *not* built on top of an
unvalidated fit -- see its section below.

## Chain solver (Step 1)

`configs/base.json`'s `solvers.default` now points at `"chain"`
(`mythos.solvers.chain.ChainSolver`, `src/mythos/solvers/chain.py`): tries each solver
in order, first non-`SolverError` result wins. `mythos.solvers.factory.make_solver`
builds it as `SymbolicSolver -> AgenticReplSolver -> FixtureSolver ->
PlannedPipelineSolver`, which is **not** the naive "verified solvers, then the
already-better neural pipeline, then a guaranteed fallback" ordering it might look
like -- `PlannedPipelineSolver` is placed *last*, not third, on purpose. Checked
before wiring it in: in this repo's default config (no `MYTHOS_ENABLE_REAL_HRM` /
`MYTHOS_ENABLE_TTT` / etc. set), every stage of `PlannedPipeline` falls back to
`BaselineSolver`'s unverified trivial guess *unconditionally* -- `strict_models` only
changes behavior when a model is enabled but then fails, not when it's simply
disabled -- so `PlannedPipelineSolver.solve()` never raises `SolverError` and would
otherwise always short-circuit the chain before `FixtureSolver`'s better, actually
train-pair-verified guess got a turn. Confirmed with
`tests/test_chain_solver.py::test_make_solver_chain_degrades_without_llama_cpp_installed`.

Also found while wiring this in: `AgenticReplSolver`'s real backend
(`LlamaCppClient`) loads its GGUF model eagerly in `__init__` and raises immediately
if `llama-cpp-python` isn't installed or no model is staged -- which is true on any
machine without the full Kaggle agentic stack, including this dev environment. Since
`"chain"` is now the *default* solver, `make_solver("chain")` catches that specific
failure and builds the chain without the agentic-REPL step rather than making the
default solver uninstantiable outside Kaggle; every remaining solver in the chain
still only ever returns a train-pair-verified prediction, so correctness is
unaffected, only coverage.

## Simplify-pass cost/benefit (Step 2)

`AgenticReplSolver` now takes a `simplify_rounds` parameter (default 1, matching the
existing `refinement_rounds` pattern): after a candidate verifies against every train
pair, it's given one extra chance to be replaced by a shorter *and still verified*
equivalent (`agentic_repl/llm/prompts.py:build_simplify_prompt`,
`agentic_repl/complexity.py:program_complexity`). An unverified simplification is
always discarded -- this project's rule against a "simplify" step ever replacing a
verified candidate with an unverified one is enforced directly in
`AgenticReplSolver._simplify_if_possible`, and covered by
`tests/test_agentic_repl.py::test_simplify_discards_broken_simplification`.

**Complexity metric used: AST node count, not character count.** Checked before
implementing: character count rewards obfuscation (renaming variables shrinks it
without simplifying anything), and CompressARC's own MDL framing
(`third_party/compress_arc`) doesn't transfer here -- it measures bits to encode
*trained network weights*, not source code. A pure DSL-primitive-call count (via
`agentic_repl/dsl/catalog.py`'s introspection) was also considered and rejected as the
*sole* metric: this project has real, documented history (see the
"DSL-nudged prompts" commits) of the model reaching for hand-written pixel loops that
use zero DSL primitives, so two such candidates would always tie at 0 and the metric
couldn't discriminate between them. AST node count avoids both problems and is the
standard proxy used in program-synthesis Occam's-razor tie-breaking when a full
probabilistic-grammar description length (as in DreamCoder) isn't available. See
`agentic_repl/complexity.py`'s module docstring for the full reasoning.

**Measured cost, on the held-out evaluation split (120 tasks), `--llm-client stub`:**

| `simplify_rounds` | fired | total LLM calls | wall-clock |
| --- | --- | --- | --- |
| 0 | 0/120 | 1080 | 3.2s |
| 1 | 0/120 | 1080 | 3.1s |

**No difference, and that's expected, not a null result to hide:** the stub client
always proposes the literal identity function regardless of `simplify_rounds` or
prompt content, so on real held-out ARC-AGI-2 tasks (almost none of which are solved
by identity) nothing ever verifies (`fired = 0`), and a pass that only runs *after* a
candidate verifies never gets a chance to engage at all. The mechanism itself --
keeping a shorter verified simplification, discarding a broken one -- is exercised
and passing in `tests/test_agentic_repl.py` (`test_simplify_keeps_shorter_verified_candidate`,
`test_simplify_discards_broken_simplification`) against scripted `FakeLLMClient`
responses, but a real cost/benefit number (does simplification measurably improve
generalization, and at what LLM-call/wall-clock cost, against tasks the *real* model
actually solves) needs the same Kaggle GPU + staged model this whole section's
decision gate is waiting on -- recorded here rather than assumed, per this project's
own measurement rule.

## Solved-program retrieval library (Step 3) -- researched, not implemented this cycle

**Not built in this cycle, deliberately.** Two independent reasons surfaced, either
one enough on its own:

1. **The decision gate above is unresolved, not passed.** This project's own rule is
   "never build on top of an unvalidated fit" -- Step 3 is specifically an
   *additional* investment on top of `agentic_repl`, and whether `agentic_repl`
   generalizes to held-out tasks at all is exactly the open question this document's
   Held-out evaluation section couldn't answer locally.
2. **The framing in this cycle's plan doesn't match what SOAR actually does.**
   Checked directly against the paper (Pourcel, Colas & Oudeyer, "Self-Improving
   Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI",
   ICML 2025 -- [arXiv:2507.14172](https://arxiv.org/abs/2507.14172)): SOAR's
   hindsight relabeling converts *failed* search attempts into valid
   `(synthetic-task, program)` pairs used **only to build a LoRA fine-tuning
   dataset** -- there is no retrieval-augmented few-shot/in-context mechanism in
   SOAR at all. A solved-program retrieval library (inject similar past solutions
   as few-shot context at inference time) is a real, independently-motivated idea
   from the general retrieval-augmented-code-generation literature -- but it is
   not "the cheap first half of SOAR's mechanism" as originally framed, and
   presenting it that way would have been a false citation.

**What the research did surface, for whenever the gate resolves and this gets
built:**
- Retrieval-augmented few-shot prompting for code has a **documented "more
  retrieval hurts" failure mode**: CEDAR (Nashid et al., ICSE 2023,
  "Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning") and a more
  recent study, "When More Retrieval Hurts: Retrieval-Augmented Code Review
  Generation" ([arXiv:2511.05302](https://arxiv.org/pdf/2511.05302)), both find
  that a single well-chosen retrieved example (top-1) often outperforms top-3/top-5
  -- more retrieved context adds redundant or conflicting cues, not signal. If this
  library is built, it should inject **one** retrieved example by default, gated by
  a similarity threshold, not an unconditional top-k -- directly contrary to this
  cycle's original strawman (`k: int = 3`).
- On the similarity representation itself: this repo already has a real trained
  embedding model (`src/mythos/jepa_encoder.py`) rather than needing a
  hand-designed structural signature from scratch -- worth checking whether it
  produces usable embeddings standalone (without the rest of the shelved HRM path)
  before defaulting to hand-picked structural features (grid-size deltas, color
  count, object count), which is the weaker of the two options per the retrieval
  literature's general preference for learned over hand-designed similarity when a
  usable embedding already exists.
- No published guidance was found on task-level vs. primitive-level retrieval
  specifically for ARC-style DSL program synthesis (i.e. whether "other tasks that
  used `fill_enclosed_regions`" beats "other tasks with a similar grid size") --
  this remains an open design question to test empirically, not something the
  literature already settled, if/when this gets built.

**To unblock:** resolve the Held-out evaluation gate above (run
`benchmark_agentic_repl.py --llm-client llamacpp` on Kaggle against the 120-task
held-out split). If the combined result holds up, build `agentic_repl/library.py`
per this cycle's original file/test layout, but with `k=1` + a similarity threshold
as the retrieval default per the finding above, and cite the retrieval-augmented-
code-generation literature (not SOAR) for that part of the design.

## Hindsight fine-tuning (Step 4) -- not started

Explicitly gated on Step 3 ("Do not start this until Steps 1-3 are built, measured,
and the library has accumulated a non-trivial number of verified solutions"). Since
Step 3 wasn't built this cycle (see above), Step 4 is doubly blocked -- not started,
and its own research-first questions (SOAR's actual LoRA recipe, catastrophic-
forgetting mitigation, GGUF re-quantization verification) weren't investigated in
depth this cycle either, beyond confirming above that the paper doesn't publish
concrete LoRA hyperparameters in its abstract/summary (the full PDF would need a
closer read before designing `scripts/finetune_agentic_llm.py`'s training recipe).

## TL;DR

- Built a real HRM (Hierarchical Reasoning Model) inference + test-time-training (TTT)
  + LoRA pipeline for the ARC Prize 2026 (ARC-AGI-2) Kaggle competition, following
  MindsAI's ARC-AGI-1-winning approach.
- Found and fixed ~15 real bugs across the TTT/LoRA training loop, HRM's dataset
  plumbing, and — most impactful — the output-shape decoding logic, which was silently
  capping exact-match accuracy at 0% for an entire task family regardless of model
  quality.
- Built a from-scratch, verified symbolic/DSL solver (symmetry repair, fragment-to-slot
  object placement, object selection, region filling) that runs ahead of the neural
  pipeline and produces genuine, verified exact matches on held-out data at zero
  marginal cost.
- Discovered via the ARC Prize organizers' own published analysis that HRM's true
  ceiling on ARC-AGI-2 is **~2%** (not the ~5% figure that circulates), with a specific
  architectural reason. This — not a bug in this codebase — is why the neural path
  never produced a verified exact match across ~9 real Kaggle validation runs.
- Investigated CompressARC (a from-scratch, no-pretraining alternative with a much
  higher published ceiling) as a replacement. Real Kaggle T4 testing found the
  per-task compute cost makes a full competition run infeasible within a single
  session without a nontrivial parallel-scheduling engineering effort with an
  uncertain payoff — shelved rather than pursued further.

## What this project was

An ARC-AGI-2 solver targeting Kaggle's "ARC Prize 2026" code competition
(`arc-prize-2026-arc-agi-2`). The core bet, inherited from an earlier project plan, was
that test-time training (TTT) with LoRA adapters on top of a pretrained HRM checkpoint
— the approach MindsAI used to win ARC-AGI-1 — would generalize to ARC-AGI-2. It does
not, for reasons documented below, and the project pivoted mid-session toward a
verified symbolic solver and an investigation of alternative architectures before
being shelved.

## Architecture as of the last commit

```
src/mythos/
  arc.py, submission.py, metrics.py      - ARC task I/O, submission format, scoring
  features.py                            - HRM token encoding/decoding (grid <-> sequence)
  objects.py, augment.py                 - connected-component segmentation, D4 grid transforms
  symmetry.py, object_ops.py             - verified symbolic-solver primitives (see below)
  lora.py, losses.py                     - LoRA adapter injection, Genie consistency loss
  hrm_dataset.py                         - glue for the external HRM repo's dataset builder
  kaggle_run.py                          - Kaggle entrypoint: solve chain, submission writer
  solvers/
    symbolic.py                          - verified solver: symmetry repair -> fragment-to-slot
                                            -> object selection -> fill-enclosed -> rigid transforms
    hrm.py                               - HRMInferenceRunner, HRMTTTRunner (per-task TTT/LoRA
                                            + inference-time dihedral-view ensembling)
    compress_arc.py                      - wrapper around vendored CompressARC (shelved, see below)
    fixture.py, baseline.py, pipeline.py - earlier/simpler solvers, still used as fallbacks

third_party/compress_arc/                - vendored CompressARC (MIT), not wired into the
                                            active solve chain -- see "The CompressARC
                                            investigation" below
scripts/
  build_standalone_notebook.py           - generates project_mythos_kaggle_pipeline_standalone.ipynb
                                            by embedding the whole src/mythos + third_party tree
  build_compress_arc_timing_notebook.py  - generates the isolated CompressARC timing-test kernel
  kaggle_ci/                             - push/watch/score automation for the Kaggle kernel
  benchmark_symbolic.py                  - local, zero-Kaggle-cost benchmark for the symbolic
                                            solver against the real public ARC-AGI-2 dataset
```

The actual Kaggle submission is a single generated notebook
(`project_mythos_kaggle_pipeline_standalone.ipynb`) that embeds the entire `src/mythos`
package and `third_party/` inline, since the competition kernel runs with internet
disabled. Regenerate it after any source change with:

```bash
python scripts/build_standalone_notebook.py
```

## Solve order (the actual pipeline)

For each task, in order, first solver to produce a prediction wins:

1. **Symbolic solver** (`mythos.solvers.symbolic.SymbolicSolver`) — tries, in order:
   symmetry repair (recovers an occluded grid region from the grid's own mirror/
   rotation/periodic symmetries), fragment-to-slot object placement (matches scattered
   colored fragments to shape-matching occluded slots, allowing rotation/reflection),
   object selection (crop to the largest/smallest/uniquely-colored/uniquely-shaped
   object), fill-enclosed-regions, then falls back to a small verified rigid-transform
   search (`fixture.py`: identity/mirror/rotate/recolor/translate). **Every candidate
   here is checked for exact agreement against every train pair before ever being
   applied to a test input** — it either returns a provably-correct-on-the-demos
   answer or raises and lets the next solver try. This is why it's safe to run first
   unconditionally.
2. **HRM + TTT** (`mythos.solvers.hrm.HRMTTTRunner`) — only for tasks the symbolic
   solver didn't solve. Loads the HRM checkpoint once, injects LoRA adapters (rank 16,
   targeting attention *and* MLP layers), then per task: resets LoRA to its no-op
   initial state, runs gradient descent against only that task's own demo pairs
   (`MYTHOS_TTT_STEPS`, default 100, with a Genie background-consistency auxiliary
   loss), and predicts. Optionally ensembles across a small set of dihedral transforms
   of the whole task (`MYTHOS_TTT_ENSEMBLE`), each getting its own from-scratch TTT
   fit, voting across the un-augmented predictions.
3. **Baseline fallback** — guaranteed-output trivial predictions, so a Kaggle rerun
   always produces a valid `submission.json` even if everything else fails on a given
   task.

## Major issues found and fixed

Roughly in the order they were found, via real Kaggle runs and real data — not
speculation:

### Infrastructure / plumbing
- Kaggle's current CLI (`kagglesdk`-based) uses different flags than older docs assume
  (`-o` not `-f`, positional competition slug not `-c`, `kernels logs` not `kernels
  output`), and its rich-table stdout crashes when piped on Windows unless
  `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` are forced.
- `default_run_dir()` was relative, resolving to two different real paths across the
  dataset-builder subprocess and the process reading the built dataset back.
- HRM's own dataset-builder script ignores `--dataset-dirs` on the CLI and always scans
  a hardcoded relative default; worked around by giving the subprocess a matching
  writable `cwd` instead of trying to override its config.

### TTT / LoRA training loop (11 bugs, each found via a real Kaggle run)
LoRA adapters created on CPU/fp32 by default (broke `torch.compile` tracing and caused
dtype-mismatched matmuls against HRM's bf16-activation/fp32-weight layers); HRM's
`_iter_train` silently drops any batch smaller than the configured batch size, which
for a while meant **TTT was running zero actual gradient steps** while looking like it
worked; the train loader yields a 3-tuple, not 2; `initial_carry()`'s internal tensors
default to CPU even inside a CUDA model; the `puzzle_emb` sparse embedding buffer is
sized once at model-init to a fixed batch size and cannot accept a different one later,
so every per-task dataset build has to match it exactly; the Genie consistency loss was
implemented but never actually invoked (gated on a condition that never became true);
logits needed reshaping from a flat `[900, vocab]` sequence into `[30,30,vocab]` before
slicing color channels; `rank=64`/`lr=3e-3` caused loss divergence, reverted to a
conservative `rank=16`/`lr=1e-4`.

### Output-shape decoding — the highest-impact fix
`output_shape_hint()` only trusted train examples' output shape when every example
agreed; when they disagreed (common for crop/symmetry-repair-style tasks, where output
size is the bounding box of some occluded region and varies per example) it silently
fell back to **the full input grid's shape**. Every prediction for that task family was
then decoded as a full grid, which can never exact-match a small patch — a hard 0%
ceiling for that whole task family regardless of model quality. Fixed to defer to the
model's own predicted EOS boundary tokens instead. The EOS-boundary detector itself
then turned out to be too fragile (fired on the *first* row/column with ≥2 EOS tokens,
but every in-grid row/column already legitimately carries one stray EOS token from the
orthogonal boundary marker, so a single unit of noise anywhere false-triggered it) —
fixed to use argmax (the strongest EOS-run) instead of first-crossing.

### Symbolic solver bugs (found via its own test suite + real-data benchmarking)
The symmetry-axis search initially assumed the mirror axis was the geometric center of
the grid; a real task showed the axis can be off-center (the grid can be a window onto
a larger symmetric pattern) — fixed by searching all axis positions and adding closure
under composition (composing two independently-verified symmetries is transitively
still a symmetry — algebra, not a heuristic — which is what discovers compound cases
like glide reflections without hand-listing every symmetry family). The
occlusion-color-candidate detector originally summed area across all same-color
components, which would let many scattered single pixels of one color (e.g. a
checkerboard) falsely qualify as "one occlusion block" — fixed to gate on the largest
single component instead.

## Key findings from research

The most important thing learned this session didn't come from our own code — it came
from checking what the field actually knows about ARC-AGI-2:

- **HRM's real, ARC-Prize-verified score on ARC-AGI-2 is 2%**, not the ~5% figure that
  gets cited. Source: [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis).
  Their own words: *"scores >0% show some signal, we do not consider this material
  progress on ARC-AGI-2."*
- The "hierarchical" architecture HRM is named for is largely irrelevant — a plain
  standard transformer with no tuning lands within 5 percentage points of it. The
  actual driver of HRM's performance is its iterative-refinement loop count (1→8 loops
  *doubles* accuracy), a lever this project never tuned.
- **Cross-task transfer is minimal**: most of HRM's reported performance comes from
  having effectively seen the specific evaluation tasks during pretraining, not from
  genuine generalization to novel puzzles.
- **The load-bearing limitation for this project specifically**: HRM "can only be
  applied on puzzles with puzzle_ids it has seen at training time" (per the same ARC
  Prize analysis). HRM has a per-puzzle embedding table; this project's LoRA target
  list never touched it, meaning every genuinely novel task got an untrained embedding
  slot with no mechanism to adapt it.
- Comparable small-model approaches, for scale: Tiny Recursive Model (TRM, 7M params,
  the field's 2025 top paper-prize winner) reaches 6.67–10% on ARC-AGI-2 using **full
  fine-tuning** (not LoRA — their own ablation found LoRA underperforms for a model
  this size), ~12,500 TTT steps at batch 384, and 256–512-view inference ensembling —
  every lever pulled far harder than this project's compute budget allowed.
  CompressARC (no pretraining at all) reaches 20–34.75%. Both numbers make clear that
  even a well-executed small-model approach tops out well below the "30–40" scores
  this project originally hoped for; that target was implicitly calibrated against
  ARC-AGI-1's difficulty, not ARC-AGI-2's.

## What actually worked (validated results, not projections)

- **The symbolic solver**: benchmarked locally against the real public ARC-AGI-2
  dataset (1000 training + 120 evaluation tasks, fetched fresh from
  `arcprize/ARC-AGI-2` on GitHub — zero Kaggle cost). Result: 25/1000 training tasks
  solved (88% of those exactly correct), 1/120 evaluation tasks solved (100% exact).
  Modest coverage (~2%), but genuine, verified, free marginal accuracy — it only ever
  fires when it has a provably-correct-on-the-demos rule, so it can never make a
  working prediction worse. Reproduce with `scripts/benchmark_symbolic.py` (fetch the
  dataset per the script's own note first).
- **TTT measurably improves cell-accuracy over the raw checkpoint**, confirmed via a
  real A/B on Kaggle: baseline (no TTT) scored 389/7507 matched cells on a 10-task
  diagnostic sample; TTT at 50 steps scored 3697/7507; at 200 steps, 4316/7507.
  Real, reproducible, and monotonically improving with more compute.
- **Inference-time ensembling gives a small but real gain**: a 2-view dihedral
  ensemble at steps=100/view (roughly matched total compute to the steps=200
  single-view baseline) scored 4375/7507 matched cells — a genuine if modest
  improvement, validated on real Kaggle T4 hardware with zero failed views.

## What didn't work

- **The HRM+TTT neural path never produced a single verified exact match** across
  every real Kaggle diagnostic run this session (roughly 9 separate validation
  kernels). This is consistent with — and now explained by — HRM's independently
  documented ~2% true ceiling on ARC-AGI-2, not a bug in this implementation.
- Scaling TTT steps further hit diminishing returns (389→3697→4316 matched cells at
  0→50→200 steps) and is resource-capped anyway: TRM needed ~12,500 steps to reach its
  numbers, 60x more than this project's budget allowed within a single Kaggle session.

## The CompressARC investigation

Given HRM's documented ceiling, [CompressARC](https://github.com/iliao2345/CompressARC)
(MIT-licensed, vendored into `third_party/compress_arc/`) was investigated as a
replacement: it trains a small, randomly-initialized network from scratch per puzzle
via a compression/MDL objective, needs no pretrained checkpoint at all, and its paper
reports 20–34.75% on ARC-AGI-2 — well above anything HRM can offer.

Real Kaggle testing (`compress_arc_experiment/`, an isolated kernel kept separate from
the main pipeline) found:
- **Per-step cost on a T4**: ~2.4–5.4s/step, vs. the paper's ~0.6–0.8s/step on a
  desktop RTX 4070. At the paper's 2000-step recipe, that's ~193 hours *sequential*
  for the full 120-task test set.
- **P100 is not usable at all**: Kaggle's current PyTorch build only ships CUDA
  kernels for compute capability sm_70+; P100 is sm_60. Every task failed instantly
  with `AcceleratorError`. TPU would require a full XLA/JAX rewrite — out of scope.
  **T4 is the only viable GPU option** for this workload on Kaggle today.
- **Real per-task GPU memory** (measured across 20 real tasks): 0.198–3.174GB, mean
  1.334GB, implying roughly 4–10x safe concurrency on a 16GB T4 depending on
  scheduling strategy. Combined with the per-step cost, a best-case parallel estimate
  lands around 10–19 hours for the full test set — plausibly close to a single Kaggle
  session's budget, but not clearly under it, and confirming the real number requires
  building CompressARC's own dynamic multiprocess scheduler
  (`third_party/compress_arc/parallel_train.py`), a genuine engineering effort with no
  way to validate short of building it.

**Decision: shelved, not pursued further.** The vendored code and real measurements
are kept (not discarded) in case parallelization is worth revisiting later with a
clearer time budget.

## Recommendations for a new architecture

Based on everything above, if returning to this problem:

1. **Don't start from HRM.** Its documented ceiling on ARC-AGI-2 is too low to be worth
   the TTT engineering investment this session made. TRM (full fine-tuning, not LoRA)
   or CompressARC (no pretraining, but needs real parallel-scheduling engineering to
   fit a Kaggle session) are the credible small-model alternatives with actual
   published numbers on this benchmark.
2. **Object-centric decomposition + verified symbolic search is real and cheap.** The
   symbolic solver here, built from scratch in a few hours, produced the only
   unambiguously-verified exact matches this project achieved, at zero marginal
   inference cost. A more complete DSL (more primitives, deeper composition search)
   is very likely the highest-leverage next investment, independent of whatever neural
   backbone is chosen.
3. **If pursuing TTT again, use full fine-tuning over LoRA** for models in the
   10–30M parameter range, per TRM's own published ablation — LoRA's low-rank
   restriction appears to cost more capacity than it saves in overfitting protection
   at this scale, though this needs pairing with heavier per-task data augmentation to
   compensate for the lost implicit regularization.
4. **Budget real GPU-hours before committing to an architecture.** Every serious
   small-model ARC-AGI-2 result in the literature uses 10–100x more TTT/training
   compute than a single Kaggle T4 session provides. Know the real per-step cost on
   the actual target hardware (not a paper's benchmark GPU) before designing around a
   step count or ensemble size.

## Setup (for reference)

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Local commands:

```bash
python -m mythos.validate data/toy/challenges.json
python -m mythos.solve --solver symbolic --challenges data/toy/challenges.json --out runs/submission.json
python -m mythos.score --pred runs/submission.json --solutions data/toy/solutions.json
python scripts/benchmark_symbolic.py --challenges <path> --solutions <path>  # needs the real ARC-AGI-2 dataset, see script docstring
```

The Kaggle submission notebook is generated, not hand-edited:

```bash
python scripts/build_standalone_notebook.py
```

Kaggle push/watch/score automation lives in `scripts/kaggle_ci/`; see
`python scripts/kaggle_ci/run_ci.py --help`.

External model checkpoints (HRM) are configured via environment variables when
running outside Kaggle's pre-staged Dataset mount:

```bash
export HRM_REPO_DIR=/path/to/HRM
export HRM_CHECKPOINT_PATH=/path/to/checkpoint.pt
```
