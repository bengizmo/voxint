# Voiceclone V2 (#664): synthetic target voice pack

Status: draft

## Goal

Voice risk reduction (V5, #667) converts a consented speaker's turns with
Chatterbox VC (MIT, pinned `5de7a54a`, weights `ResembleAI/chatterbox@5bb1f6ee`)
into a synthetic target voice. ADR 0009 section 4 requires those targets to be a
vendored pack of synthetic voices, shipped as a sha-pinned release asset with
provenance, and never a real third party's voice.

V2 produces that pack:
- four voices designed from text descriptions;
- published once as the immutable GitHub release `voiceclone-target-voices-v1`;
- with tool-written provenance, a committed sha manifest, a screening report, a contract test,
  and a doc line naming the voices as synthetic.

Released assets are immutable. A mistake means `-v2`.

## Decisions already made (2026-09-30, settled)

- **Generator: Qwen3-TTS-12Hz-1.7B-VoiceDesign.**
  - Code: Apache-2.0, `QwenLM/Qwen3-TTS` at `022e286b98fbec7e1e916cb940cdf532cd9f488e`, which
    has a LICENSE file.
  - Weights: HF revision `5ecdb67327fd37bb2e042aab12ff7391903235d3`. The licence is the card
    metadata `license: apache-2.0`; the HF repo has no LICENSE file.
  - Package: PyPI `qwen-tts==0.1.1`, which hard-pins `transformers==4.57.3` and
    `accelerate==1.12.0`.
  - Weight sha256s: `model.safetensors` `391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522`,
    `speech_tokenizer/model.safetensors` `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
  - Voices come from text descriptions only, never from a recording.
- **Chatterbox's built-in `conds.pt` voice is ruled out as a target.** Its provenance is
  undocumented, and the upstream README says its prompts are "sourced from freely available data
  on the internet".
- **Four voices:** two lower and two higher in perceived pitch, spread across the range.
- **Scope:** V2 stops at the asset, provenance, sha manifest, screening report, contract test and
  doc line. V3 (#665) does the Dockerfile `ARG` + `sha256sum -c` wiring and the `release.yml`
  `VOICECLONE_TARGET_VOICES_RELEASE` env.
- **Screening:** each candidate is checked with Voxint's own TitaNet against real speakers and
  against the other candidates. A real-speaker cosine at or above the 0.60 match floor rejects
  the candidate.

## Assumptions and constraints

- **Target reference format.** From `vc.py` and `s3gen.py` at `5de7a54a`:
  - The reference is loaded with `librosa.load(sr=24000)`, which mixes it down to mono.
  - It is truncated to `DEC_COND_LEN = 240000` samples (10 s).
  - A 16 kHz copy feeds the x-vector and the speech tokens (25 per second).
  - When the length is not a multiple of 960 samples (40 ms), the mel and token lengths
    disagree, and VC warns and trims.

  Qwen3-TTS outputs 24 kHz natively, so the asset needs no resample.
- **The 0.60 floor is in-domain for this screen.** Voxint's matcher (`MatchingGates.min_cosine`,
  `src/voxint/speakers/matching.py:85`) compares a label centroid against roster centroids. A
  roughly 10 s candidate clip scored against a real speaker's centroid is that same comparison.
  Clip-to-clip maxima are reported as well, as the more conservative view.
- **Qwen3-TTS has no seed argument.** Sampling is on by default (temperature 0.9, top_k 50,
  top_p 1.0, repetition_penalty 1.05). The tool sets `torch.manual_seed` itself. Determinism is
  not promised across hardware. The WAV sha is the artifact; seed, parameters, device, dtype and
  versions are provenance, not a reproducibility guarantee, and the README says so.
- **Hardware.** The official path is CUDA with bf16, and flash-attn is optional. MPS is
  unsupported, and fp16 produces NaNs (the maintainer recommends fp32 where bf16 is
  unavailable).
  - Decision: all candidates are generated on one rented RTX 3090 with bf16. That is the
    maintainer-tested configuration, about $0.22/h.
  - The VC load smoke and the usefulness matrix run in the same session with the existing Gate 0
    Chatterbox image (torch 2.8 cu128 supports sm_86).
  - The pack never mixes devices or dtypes.
- **No bloat.** The generator stack never enters Voxint's `pyproject.toml` or `uv.lock`. The
  generator is a standalone PEP 723 script with exact inline pins. The torch pin is fixed after
  the slice-1 spike and before bulk generation.
- **Screening limits, stated publicly.**
  - Qwen's training data (over 5 million hours, undisclosed) cannot be screened.
  - The evidence that a voice is synthetic is the process: an attribute-only text description,
    with no reference recording.
  - The screen is a collision check against a named pool, in Voxint's own embedding space. It is
    not proof of non-resemblance to anyone, and the public wording never claims it is.
- **Outputs.** Apache-2.0 places no restriction on outputs. Precedent:
  `tools/generate_tutorial_audio.py` dedicates project-generated audio to CC0.
- No living spec is declared in `CLAUDE.md`.

## Proposed approach

1. **Spec, committed first:** `tools/voiceclone_target_voices.spec.json`.
   - Voice ids are neutral (`target-a` to `target-d`), never given names.
   - Each voice has an attribute-only description: gender presentation, age band, pitch, pace,
     timbre, delivery. No names, no "sounds like".
   - All voices read one shared project-authored passage, phonetically varied and sized for about
     11 s of speech, so the targets differ in voice rather than content.
   - Seed list: 5 per slot.
   - Sampling parameters.
   - Every DSP constant:
     - peak normalisation to -1.0 dBFS;
     - the silence-trim energy threshold and window;
     - the minimum length, 8.0 s;
     - the maximum length, 240000 samples;
     - alignment to a multiple of 960;
     - PCM16 output, no dither.
   - A prompt change is provenance-material and goes through review.
2. **Generator:** `tools/generate_voiceclone_target_voices.py`, a PEP 723 script modelled on
   `tools/generate_tutorial_audio.py`.
   - Before loading, it verifies the HF snapshot weight sha256s and fails closed. It then sets
     `HF_HUB_OFFLINE=1`.
   - For each voice and seed:
     - generate the audio;
     - apply the spec's post-processing;
     - write a 24 kHz mono PCM16 WAV;
     - hash it at write time.
   - A post-processed candidate shorter than 8.0 s is rejected, not padded.
   - It writes a per-candidate JSON:
     - the WAV sha256;
     - format fields read back from the written WAV bytes, not from intended parameters;
     - median F0 (pyin);
     - seed, parameters, device, GPU model, driver and CUDA version, dtype, attention
       implementation, library versions and the tool's git sha.
3. **Listen pass.** A fixed rubric, recorded per candidate by sha:
   - clicks, metallic artefacts, phoneme collapse, language drift;
   - whether the voice matches its description;
   - naturalness on a 1-5 scale.

   Ben listens; a second listener is optional. Shortlist at least two per slot. A final
   "tell all four apart" check runs on the exact shipped bytes.
4. **Screen tool:** `tools/screen_voiceclone_target_voices.py`.
   - It runs in the Voxint dev venv with no new dependencies. It consumes candidates by sha and
     refuses a file whose sha differs.
   - It embeds through the shipped titanet ONNX engine (`titanet-onnx-v1`, titanet-large-v2
     space) after the pipeline's own 24 to 16 kHz resample. The engine id and the resampler are
     recorded.
   - Real-speaker pool:
     - the 15 AMI and VoxConverse speakers from the V1 Gate 1 refset (180 clips);
     - plus LibriTTS-R `test-clean` (CC-BY-4.0, 39 speakers).

     A committed pool manifest pins the dataset versions and subsets, and each clip's id and
     sha256. Pool audio is never committed or released.
   - It writes `screening_report.json`:
     - the max cosine to any speaker centroid, and to any single clip, each naming the speaker;
     - headroom to 0.60;
     - each candidate's percentile within the real cross-speaker cosine distribution, as a null
       baseline;
     - the full pairwise matrix of the shortlisted voices.
   - Rules:
     - **Reject** any candidate with a centroid or clip max at or above 0.60.
     - Treat a pass with headroom under 0.05 as a swap candidate.
     - Choose finalists to maximise the minimum pairwise distance, with F0 confirming the
       pitch spread.
     - The pairwise values are reported, not used as a pass rule. Cross-speaker pairs normally
       sit far below 0.60, so it would be a near-empty bar.
5. **Chatterbox checks** (same rented GPU session, Gate 0 image).
   - **Hard:** each finalist loads as `target_voice_path` and converts at least one source with
     no exception and no length-mismatch or truncation warning.
   - **Pre-registered swap rule:** convert the 15 Gate 1 held-out sources onto each finalist and
     measure target cosine with TitaNet.
     - A slot reached by fewer sources at 0.60 than the built-in voice managed in Gate 1 (11 of
       15) is swapped for its next shortlisted candidate.
     - This rule is fixed before the run.
     - The full 15 x 4 matrix is published in the PR whatever the outcome.
   - Iteration protocol:
     - When a slot runs out of passing candidates, generate 5 more seeds.
     - After two seed rounds fail, revise the prompt (a provenance change, re-reviewed).
     - After two prompt revisions fail, stop and ask Ben.
6. **Assembler:** `--assemble` mode in the generator, or a small sibling script.
   - It merges the per-candidate generation records, the screening report and the Chatterbox
     results into `services/voiceclone/target_voices/provenance.json` and `SHA256SUMS`.
   - Humans edit only the spec. Nothing in provenance is hand-transcribed.
   - Provenance also carries:
     - `schema_version`;
     - the release tag;
     - the asset licence and the generator attribution;
     - the HF card URL, revision, licence string and date checked;
     - `reference_recording_used: false`;
     - the screening-limits statement.
7. **Docs.**
   - `services/voiceclone/target_voices/README.md` (technical) covers:
     - what the voices are;
     - how they were made and screened;
     - the limits;
     - "any resemblance to a real person is coincidental";
     - the remediation policy: a credible resemblance claim ships a `-v2` pack.
   - The release notes carry the same synthetic-provenance and limits wording. They follow a
     fixed template with no internal machine names.
   - CHANGELOG `[Unreleased]` `### Added`.
   - Lay-reader guide line: see Q2.
8. **Publish.**
   - Before `gh release create`, run `sha256sum -c SHA256SUMS` over the exact upload set and
     confirm the set matches the manifest exactly (no extras).
   - Upload the four WAVs, `provenance.json`, `SHA256SUMS` and `LICENSE-ASSETS.txt`.
     `LICENSE-ASSETS.txt` holds the CC0 text scoped to the four WAVs, plus the factual generator
     attribution and the training-data note.
   - Then download anonymously (`curl -L`, no auth) and run `sha256sum -c`. Paste that output into
     the PR as a required checklist item.
9. **Contract test:** `tests/contracts/test_voiceclone_target_voices.py`. It runs offline, in CI.
   - `schema_version` is present.
   - There are exactly four voices with neutral ids.
   - Each sha is 64-hex and equals its `SHA256SUMS` line. `SHA256SUMS` has exactly those lines
     and no extras. Mutating either side fails the test, naming the voice.
   - Format fields: 24000 Hz, mono, PCM16, 8.0 s to 240000 samples, and divisible by 960.
   - Generator pins: repo commit, HF revision, weight shas, package versions and SPDX ids.
   - `reference_recording_used` is false.
   - Screening: every centroid and clip max is below 0.60, and the pool descriptor and the
     titanet engine id are present.
   - Chatterbox load smoke: passed.
   - The release tag equals `voiceclone-target-voices-v1`.
   - An opt-in network test (`VOXINT_NETWORK_TESTS=1`, skipped by default and in CI) downloads the
     release anonymously, verifies the shas, and reads the format back from the actual WAV
     headers.

### Alternatives rejected

- **Commit the WAVs to git** (as the tutorial clip is). #664 names a release asset, and
  model-adjacent binaries ship as immutable releases with provenance.
- **Ship Chatterbox conditionals (`conds.pt`-style) instead of WAVs.** That couples the asset to
  engine internals, and nobody can audit it by ear. WAVs are engine-neutral.
- **Generate on MacBook CPU in fp32.** It's an untested path for a 1.7B model and slow. The
  Chatterbox checks need a CUDA GPU anyway, so one rented session covers both.
- **Pairwise distance under 0.60 as an acceptance rule.** Nearly empty (see step 4). It is
  reported and optimised, not gated.
- **Recompute the screen inside a test.** The pool audio is not redistributable in the repo, and
  re-embedding needs a running titanet service. Instead the report is tool-written, the pool is
  pinned by manifest, and the screen tool can be re-run by a maintainer.

## Acceptance criteria

This is asset and ops work, so the criteria are invariants and checks.

- **A1 Anonymous download.** Fetching the release without credentials returns every file, and
  `sha256sum -c SHA256SUMS` passes. The pasted output is in the PR, and the opt-in network test
  passes.
- **A2 Sha pinning.** Mutating any sha in `provenance.json` or `SHA256SUMS`, or adding an extra
  `SHA256SUMS` line, fails the contract test.
- **A3 Format.** Every WAV is 24 kHz mono PCM16, at least 8.0 s, at most 240000 samples, and
  divisible by 960. The values are read back from the bytes by the tool and asserted by the
  contract test; the network test confirms them from the headers.
- **A4 Provenance.**
  - Per voice: generator commit, HF revision, weight sha256s, package versions,
    device/GPU/driver/dtype, description, text, seed, sampling parameters, DSP constants and
    median F0.
  - `reference_recording_used: false`.
  - All of it is tool-written.
- **A5 Screening.** Every voice's maximum cosine is below 0.60 against both the centroid and any
  single clip of each pinned-pool speaker. Headroom, null percentiles and the pairwise matrix are
  recorded.
- **A6 Chatterbox usability.** Every voice loads and converts with no warning, and the
  pre-registered swap rule held.
- **A7 Honest wording.** The README, the release notes and the doc line say three things: the
  voices were designed from text descriptions; no recording of anyone was used; they were
  screened against a named public pool as a sanity check. They state that any resemblance is
  coincidental. None of them claims non-resemblance as established.
- **A8 Licence.** CC0-1.0 scoped to the WAVs, plus the generator attribution, ships in the
  release and in provenance.
- **Gates:**
  - ruff, mypy, unit and contracts;
  - `gitleaks dir .` and `gitleaks git .` clean;
  - no internal machine names in committed files, provenance or release notes;
  - full panel review of the PR diff before `gh release create`.

Spec deltas: none (no living spec declared).

## Affected files

| File | Change |
|---|---|
| `tools/voiceclone_target_voices.spec.json` | New. Prompts, text, seeds, sampling and DSP constants. |
| `tools/generate_voiceclone_target_voices.py` | New. PEP 723 generator, post-processing and assembler. |
| `tools/screen_voiceclone_target_voices.py` | New. TitaNet screen, null baseline and pairwise matrix. |
| `tools/voiceclone_screening_pool.json` | New. Pinned pool manifest: dataset versions and clip ids + sha256. |
| `services/voiceclone/target_voices/provenance.json` | New, tool-written. |
| `services/voiceclone/target_voices/SHA256SUMS` | New. V3's Dockerfile will `sha256sum -c` against it. |
| `services/voiceclone/target_voices/screening_report.json` | New, tool-written. |
| `services/voiceclone/target_voices/README.md` | New, technical lane. |
| `tests/contracts/test_voiceclone_target_voices.py` | New. |
| `tests/unit/test_voiceclone_target_voice_tools.py` | New. Pure post-processing and screening maths, including full-scale and boundary cases, empty or degenerate centroids and all-silence input; at least 85 % coverage. Plus one fixture-backed screen run with known cosines. |
| `CHANGELOG.md` | `### Added` entry. |

## Implementation slices

1. **Spec + pure functions (TDD).**
   - Spec file, post-processing and screening maths, with unit tests.
   - The pool manifest is built and the real-speaker centroids are embedded locally.
   - The null distribution is computed.
   - The tree stays green.
2. **Rented-GPU spike, then bulk generation.**
   - Generate one voice and one seed, then lock the torch/CUDA pins in the script.
   - Generate 4 slots x 5 seeds, each hashed at write.
   - Pull the results back by sha.
3. **Listen + screen + select.** Rubric pass, screening report and finalist choice by sha.
4. **Chatterbox checks** (rented GPU, Gate 0 image): load smoke, the 15 x 4 matrix, the
   pre-registered swap rule, and iteration if needed.
5. **Assemble + contract test + docs**, on branch `feat/664-target-voice-pack`.
   - Full panel review of the diff.
   - Final listen on the shipped bytes.
6. **Publish.**
   - Release with the pre-upload `sha256sum -c`; anonymous-download proof; network test.
   - Push both remotes; open the PR with "Closes #664"; CI green; merge; sync Forgejo.
   - Comment on #664 and #672.

## Testing strategy

- **Unit:** trim and alignment boundaries, the -1 dBFS ceiling, rejection under 8.0 s, centroid
  maths (degenerate inputs included), percentile and baseline maths, and the assembler schema.
- **Fixture integration:** the screen tool against small synthetic embeddings with known
  cosines.
- **Contract (CI, offline):** A2, A3, A4, A5 (structure and values), A6 (recorded result), A8.
- **Opt-in network:** A1 and A3 from the real headers.
- **Manual, recorded in the PR:** listen rubric, screening table, 15 x 4 VC matrix,
  anonymous-download output.

## Risks and open questions

- **Q1 (decided 2026-09-30): CC0-1.0** scoped to the four WAVs, with factual Qwen attribution
  and a training-data note, matching the tutorial-audio precedent.
- **Q2 (decided 2026-09-30):** the vetted line goes in the README, the release notes and the
  CHANGELOG in V2. V8 (#670) lifts it verbatim into the lay-reader guide. #664 gets a comment
  saying the guide item carries to V8. No `docs/how-to/` stub.
- **Q3 (decided 2026-09-30):** one rented RTX 3090 session (bf16) for generation and the
  Chatterbox checks. The pod is torn down afterwards.
- **Risk: training-data resemblance.** It can't be excluded. Mitigations: attribute-only
  prompts, the screen, honest wording, and the `-v2` remediation policy.
- **Risk: the weights licence rests on card metadata only.** The card URL, revision, string and
  date are recorded.
- **Risk: screen numbers depend on the embedder version.** The titanet engine and resampler are
  pinned in the report, and headroom is recorded.
- **Risk: chain of custody across machines.** Every step consumes files by sha, and the
  pre-upload `sha256sum -c` gates the release.

## Review notes

Panel, 2026-09-30: deepseek-v4-pro, grok-4.5, glm-5.2 and kimi-k3, called directly. Codex was
out of credits until 2026-10-03. Each critiqued the draft; agreement counts are in brackets.

- **The claim outruns the evidence [4/4].** The draft's goal and doc line leaned on "belongs to
  no real person", which a 54-speaker screen can't support. *Accepted:* the public wording rests
  on the process (text-only design, no recording), names the screen as a sanity check, adds
  "any resemblance is coincidental" and the `-v2` remediation policy (A7; kimi L9).
- **Centroid-only screening misses clip-level near-hits [3/4: glm, kimi, grok].** *Accepted:*
  centroid and clip maxima are both gated. Kimi's further claim, that applying 0.60 to centroid
  scores is a category error, was *checked and rejected*: Voxint's matcher compares centroids
  against roster centroids (`matching.py:85`), so the screen is in its calibration domain.
- **0.60 means little without a baseline; pairwise under 0.60 is a near-empty bar [4/4].**
  *Accepted:* null cross-speaker percentiles, headroom, and pairwise values reported and
  optimised rather than gated. A larger pool such as VoxCeleb (deepseek, kimi, grok as optional)
  was *rejected for V2* because of its access terms, and the pool limits are stated instead.
- **Provenance must be tool-written, with format read back from the bytes [3/4: kimi, grok,
  glm].** *Accepted:* assembler; humans edit only the spec.
- **Pin the screening pool [3/4: deepseek, grok, kimi].** *Accepted:* pool manifest with clip
  shas; pool audio never committed.
- **Chain of custody by hash [2/4: grok, kimi].** *Accepted:* hash at write, consume by sha,
  pre-upload `sha256sum -c`, exact-set check.
- **One device and dtype for the whole pack [3/4].** *Accepted.*
- **Q3, device [3 for a rented 3090 with bf16: deepseek, glm, kimi; grok said spike both].**
  *Accepted:* rented 3090, approved 2026-09-30. The Chatterbox checks need CUDA anyway.
- **Q4, the usefulness check:**
  - *The split:* deepseek wanted a hard gate at 8 of 15, glm a hard gate at 11 of 15, kimi a
    pre-registered one-time swap rule at the Gate 1 baseline of 11 of 15, and grok no fraction
    gate, only a hard load smoke and relative outlier rejection.
  - *Resolved:* a hard load smoke (grok) plus kimi's swap rule fixed before the run, with the full
    matrix published. This is not a standing gate, because reachability depends on the source
    (Gate 1).
- **Pin the DSP constants in the spec; normalise to -1 dBFS [3/4: grok, glm, kimi].**
  *Accepted.*
- **Reject clips that come out too short, and define how to iterate when a slot fails [3/4].**
  *Accepted:* rejection under 8.0 s, plus the seed, then prompt, then ask-Ben protocol.
- **A listening rubric, a "tell them apart" check, and a final listen on the shipped bytes [3/4:
  grok, kimi, deepseek].** *Accepted.* A second listener is optional.
- **Lock the torch pin before bulk generation [2/4].** *Accepted:* end of the slice-2 spike.
- **Record the titanet engine and resampler, and the headroom [kimi].** *Accepted.*
- **Neutral ids and median F0 per voice [kimi, grok].** *Accepted.*
- **Generate longer and keep the best window [grok].** *Partly accepted:* text sized for about
  11 s, cut to the 960-aligned maximum.
- **Second embedder [kimi: Chatterbox x-vector; grok: optional ECAPA].** *Deferred.* The
  single-embedder limit is stated in the README.
- **A refusal list for celebrity prompts in the tool [grok].** *Rejected* as bloat: the prompts
  live in a reviewed, committed spec.
- **Recompute the screen in a test [grok].** *Rejected,* see Alternatives.
- **A training-data note in `LICENSE-ASSETS.txt` [deepseek].** *Accepted.*
- **A release-notes template; A1 as a required PR checklist item [grok, kimi].** *Accepted.*
- **Q1 licence [4/4 for CC0].** Decided CC0-1.0.
- **Q2 doc location [2 for README + release notes only: kimi, glm; 2 for adding a how-to stub:
  grok, deepseek].** Decided: README + release notes + CHANGELOG; guide line carried to V8.
