# Local-LLM qualification for bundled enrichment (#66)

**Date:** 2026-08-18 · **Status:** ⛔ **No candidate qualified** as an
unrestricted bundled default. **Direction chosen: a scoped Qwen bundle** —
Qwen3-4B-Instruct-2507 powers transcript enhancement + run-asset summary/entities
only, with research and the LLM name-pass left to BYO (see Recommendation for #67).

## What this measures

#67 aims to bundle an optional, CPU-only local LLM so a non-technical operator
gets working transcript enhancement + enrichment (names, run-assets, agentic
research) out of the box, with no external API key, while keeping Voxint
Apache-2.0-clean. Before bundling, a candidate must be **qualified against
Voxint's real production prompts** — not benchmarks. This report is that verdict.

The qualification drives Voxint's **unmodified** production code
(`HttpLLMClient.enhance_segments`, the run-asset producer's `generate_payload`,
and `run_research_loop` with its deterministic offline web seams) against a
frozen, hand-annotated corpus (`tests/fixtures/llm_qual/`, 19 fixtures) via a
committed harness (`tools/qualify_local_llm.py`). Six gates — structural
validity, faithfulness, semantic usefulness, grounding, latency, bounded-failure
— and every threshold were **frozen before any model output was seen**
(`tests/fixtures/llm_qual/manifest.json`). Pass policy is per-fixture across ≥3
repetitions (temperature=0 is not determinism), never an aggregate rate.

**Device method:** capability gates are device-independent (same weights, same
quant, temperature=0), so candidate *selection* was run on a discrete GPU for
speed; only latency + memory footprint — which are device-specific — are reported
from the qual host's **CPU** (16 threads), the floor #67 must guarantee for
operators with no GPU.

## Candidates

| | Granite (primary) | Qwen (fallback) |
|---|---|---|
| Model | IBM Granite 4.0 H-Tiny | Qwen3-4B-Instruct-2507 |
| License | Apache-2.0 | Apache-2.0 |
| Quant | Q5_K_M (4.95 GB) | Q5_K_M (2.89 GB) |
| GGUF sha256 | `28f5214c…1b038bc1` | `5bde5e9d…658b4ecb` |

Serving engine (the #67 CPU ship artifact): `ghcr.io/ggml-org/llama.cpp:server`
@ `sha256:092d1291…e12625`, `-c 32768 -np 1 -t 16 -fa on -ctk q8_0 -ctv q8_0
--jinja --reasoning off`, effective chat-template sha256 recorded per candidate.

## Results

**Granite 4.0 H-Tiny — 8–9 / 19.** **Qwen3-4B-Instruct-2507 — 9 / 19.** Neither
clears the bar. They fail *differently*:

| Job | Granite | Qwen |
|---|---|---|
| Enhancement (faithfulness) | ❌ **obeys prompt injection** (translated operator transcript to French on command); drops filler words; drops a diacritic; double-emits JSON | ✅ preserves wording, diacritics, and **resists injection**; ❌ still translates non-Latin script (农业→"Economic") |
| Names (self/other) | ✅ self + false-positive control; ❌ misattributes an introduction | ❌ misattributes an introduction **and** over-attributes a "named-after" decoy |
| Run-assets (summary/topics/entities) | ✅ summary + entities (grounded, decoys excluded); ❌ malformed-JSON topics | ✅ summary + entities; ❌ topics (missed the central label) |
| Research (agentic loop) | ❌ invents a URL instead of reading results, then gives up; malformed JSON | ❌ **concludes "not found" with zero investigation** (no tool use at all) |

### The disqualifier
Granite **followed** an injected instruction embedded in transcript text —
translating and replacing operator content. No server-side guardrail (`--n-predict`
cap, JSON grammar) can prevent this, because it happens inside otherwise-valid
JSON. That behavior alone rules Granite out as a bundled *default* over
untrusted transcript text. Qwen resisted the same injection.

### Shared small-model limits
Both translate non-Latin script (a faithfulness failure), both misattribute
speaker introductions, and both are weak at the multi-round agentic **research**
loop (Granite invents URLs; Qwen refuses to investigate). These are capability
limits at the 4B / tiny-MoE scale, not prompt bugs.

### Two corpus caveats (reported in the models' favor) — now corrected
Two enhancement fixtures encoded faithfulness gold more strictly than the gate's
own definition and produced *spurious* fails: `asr_errors` (a faithful
under-correction was rejected because the gold demanded the full correction) and
`multi_speaker_swap` (authorized hyphenation "ninety two"→"ninety-two" tripped an
exact-substring protected token). No cross-segment swap or corruption occurred in
either. **Both are now fixed in the committed corpus** — `asr_errors` moves to an
authorized-edit *subset* gold model (a conservative under-correction passes; only
an unauthorized change fails) and protected-token matching now tolerates authorized
hyphenation. The per-fixture counts above were measured against the pre-fix corpus;
the fixes add ~1–2 passes per model and **do not change the verdict** (neither
candidate passes). They are not re-scored here because the verdict is unchanged;
the corrected corpus is what #67 re-runs against.

## Latency & resource floor (CPU, Granite Q5_K_M reference)

- **Headline (the identified risk):** the worst-case **48k-char run-asset job
  completed in ~175 s** (median ~36 s) — **under** the 300 s budget on the qual
  host. Cold-cache first requests pay the full system-prompt prefill; warm reps
  are far cheaper.
- Prompt processing ~80–100 tok/s; single-stream decode ~5.4 tok/s; ~4.44
  chars/token on transcript English (48k chars ≈ 10.8k tokens; `-c 32768` holds
  with headroom).
- **Memory:** cgroup peak ~2.4 GB **plus** the ~4.95 GB mmap'd weights (page
  cache) — plan for **≥8 GB RAM** for the LLM container. **Disk:** 0.85 GB image
  + 4.95 GB weights.

⚠ **Deadline semantics (correcting the issue's assumption):** `LLM_TIMEOUT_SECONDS`
is **per HTTP attempt**, not per run (retries × many batches can exceed it), and
`RESEARCH_DEADLINE_SECONDS` is checked **between rounds** and cannot interrupt an
in-flight call. Voxint offers **no hard whole-job 300 s guarantee**; the gate is
worded as observed end-to-end completion.

## Serving-profile note
`--n-predict 4096` is the derived guarded cap (a full 12k-char enhancement batch
≈ 2.7k tokens + margin). ⚠ The JSON-object **grammar** sub-profile
(`--json-schema`) **failed to initialize** on the pinned image
(`common_sampler_init: error initializing grammar sampler`), so that guard is
not usable as-is — and it would not have helped, since the decisive failures
(injection obedience, translation, misattribution) occur inside valid JSON.

## Limitations — what a better serving profile could and could not move

The gates run at **`temperature=0` (greedy)** with Voxint's *unmodified* production
prompts and its bespoke JSON protocols. That is the right basis for qualifying a
reproducible *default*, but it is **off-profile for Qwen** (the model card only ever
recommends sampling — `Temperature=0.7, TopP=0.8, TopK=20, MinP=0` — and never
greedy) and it exercises the model through Voxint's own code, not the model's
intended integration. So two classes of the recorded failures must be read with
care:

- **Real model limits no harness change fixes:** Granite's injection obedience
  (deterministic instruction-following inside valid JSON), both models' speaker-
  introduction misattribution, and both models' non-Latin **translation** (a
  faithfulness failure the model elects; sampling would not make it stop). These
  stand.
- **Serving-profile / protocol confounders (weak evidence of incapability):** the
  Qwen **research-loop** failure ("concludes not-found with zero tool use") is
  measured through Voxint's bespoke strict-JSON-in-content loop, **not** Qwen's
  documented tool-calling path (Qwen-Agent's tool-call templates/parsers); Qwen is
  otherwise strong at native function-calling for its size. Likewise the malformed-
  JSON topics and the *single flaky* injection rep are consistent with greedy
  degradation. These live in jobs (**research, LLM name-pass**) that the scoped
  path leaves to BYO, so they do not move the scoped decision — but the report does
  **not** treat them as proof the model cannot do the work.

Net: harness/serving improvements would **not** rescue either model as an
*unrestricted* default (the disqualifiers are real), but they would likely improve
the agentic/format numbers — in the out-of-scope jobs. The in-scope jobs Qwen
already passes.

## Recommendation for #67

1. **Do not bundle either model as an unrestricted default.** Granite is out
   (injection obedience). Qwen is the stronger candidate but does not pass the
   full bar.
2. **Chosen direction — scoped Qwen bundle.** #67 bundles
   **Qwen3-4B-Instruct-2507** to power only the jobs it does well — **transcript
   enhancement + run-asset summary/entities** — and leaves **agentic research and
   the LLM name-attribution pass** to a BYO-endpoint (their current state). This
   ships real out-of-box value without promising capabilities a tiny local model
   cannot deliver. It requires exposing **per-job LLM enablement** so the bundled
   model powers only its scoped jobs.
3. **#67 acceptance work (against this same frozen, now-corrected corpus):**
   - **Decide and pin the shipped sampling profile.** Qwen recommends *against*
     greedy (`Temperature=0.7, TopP=0.8, TopK=20, MinP=0`); the qual measured
     greedy for a reproducible pass policy. #67 must choose the shipped sampling
     and re-measure the in-scope gates under it.
   - **Add a language-preservation instruction** to the enhancement prompt
     ("preserve the source language verbatim; never translate") to close the one
     real in-scope faithfulness fail (non-Latin translation), then re-run.
   - **CPU-benchmark Qwen** for the deadline gate (only the shipped model needs
     the CPU latency/RAM floor; Qwen's 2.89 GB Q5 is lighter than Granite's 4.95 GB).
   - Re-characterize the flaky `prompt_injection` rep with more reps.
4. **Alternatives not taken (recorded):** a larger local candidate (raises the CPU
   latency/RAM floor — must be re-qualified with this same harness), or the
   status-quo BYO-only posture. The scoped-Qwen path was selected over both.
5. **#67 must consume a model _and_ a serving profile** (engine image digest,
   GGUF + sha256, ctx, KV types, sampling, `--reasoning off`, output cap,
   chat-template hash) — not just weights. Re-run this frozen harness against any
   new candidate; never tune the fixtures or thresholds per model.

## Reproduction
`uv run python tools/qualify_local_llm.py --reps 3` against an OpenAI-compatible
endpoint serving the candidate (`--base-url/--model/--api-key`). Corpus + frozen
gate contract: `tests/fixtures/llm_qual/`.
