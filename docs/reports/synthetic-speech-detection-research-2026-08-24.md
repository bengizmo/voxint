# Adding an AI Speech Detector to Voxint: Open-Source Options, the Competitive Landscape, and an Integration Plan

> **Status:** Background research, 2026-08-24. A survey of open-source synthetic-speech detectors, the commercial landscape, watermarking, and an integration sketch for Voxint. It informs the synthdetect epic (#143) and the M1 eval harness (#144). This is analysis and options, not a commitment to any specific model, vendor, or ship date.

## TL;DR
- **Build it on an open-source SSL detector (wav2vec2/XLS-R/WavLM front-end + AASIST or Nes2Net back-end), run per-diarized-speaker-turn inside your existing Celery/Docker/GPU pipeline, and treat the output as a calibrated probability, not a verdict.** The strongest permissively-licensed starting points are Tak's **SSL_Anti-spoofing (W2V2-AASIST, MIT)** and **Liu's Nes2Net** (best accuracy/size ratio, but no license file, so contact the author); NII's **AntiDeepfake XLS-R** models are the most accurate out-of-the-box but their weights are **non-commercial (CC BY-NC-SA 4.0)**.
- **Watermark detection is a useful but narrow add-on, not the core.** Only some providers watermark (Google/DeepMind SynthID for Lyria/NotebookLM and now partners like ElevenLabs and OpenAI; Meta AudioSeal on its own products). Meta's **AudioSeal detector is fully open (MIT)** and cheap to run, so add it as a fast pre-check. The vast majority of synthetic speech in the wild carries no detectable watermark, so passive and statistical detection must do the heavy lifting.
- **Calibrate expectations: lab EERs of 1–3% collapse on real-world audio.** The independent *Deepfake-Eval-2024* benchmark found audio detectors lost an average **48% AUC** on in-the-wild audio (off-the-shelf open-source audio models peaked at only ~0.58 AUC); compression, telephony codecs, background music and unseen 2024–2026 TTS systems are the dominant failure modes. Benchmark on your own domain data, plan to re-train/refresh periodically, and surface scores as "likely synthetic" flags for human review rather than automated decisions.

## Key Findings

**1. The dominant open-source architecture is an SSL front-end + lightweight back-end.** The field has converged on self-supervised speech models (wav2vec2, XLS-R, WavLM, HuBERT, MMS) as frozen or fine-tuned feature extractors, feeding a graph-attention (AASIST) or nested-residual (Nes2Net) classifier. These consistently beat handcrafted-feature and pure raw-waveform models (RawNet2, original AASIST) on cross-dataset tests.

**2. Best open-source options for Voxint, ranked by fit:**
- **NII AntiDeepfake (xls-r/wav2vec2/MMS families)**: best raw generalization (XLS-R-2B: **1.23% EER on In-the-Wild** zero-shot, with 2.23% on DEEP-VOICE and 4.67% on ADD 2023), trained on 56k hours real + 18k hours fake in 100+ languages, HuggingFace-hosted, arbitrary-length 16 kHz input, binary real/fake output. **Code is BSD-3-Clause but model weights are CC BY-NC-SA 4.0 (non-commercial)**, the key catch for any commercial use.
- **Tak SSL_Anti-spoofing (W2V2-AASIST / SSL-AASIST)**: the canonical wav2vec2-XLS-R + AASIST model, **MIT license (code + weights)**, ~2.85% EER on ASVspoof 2021 DF, ~10.5% EER on In-the-Wild (author condition; worse under channel mismatch). 318M params. Widely used as the reference baseline (e.g., RADAR 2026 challenge).
- **Liu Nes2Net (wav2vec2-Nes2Net-X)**: state-of-the-art accuracy-per-parameter (back-end ~511k params; ~5.2–5.8% EER In-the-Wild, ~1.49% EER ASVspoof 2021 DF). **No license file present** in either repo (effectively all-rights-reserved, so contact the author before deploying).
- Supporting infrastructure: **DeepFense** (modular YAML framework, 455+ pretrained checkpoints, swappable front/back-ends), **clovaai/aasist** (original AASIST, MIT), and various HuggingFace wav2vec2 fine-tunes (mixed quality; many report inflated in-domain accuracy).

**3. Watermarking covers only a fraction of generators.** Meta **AudioSeal** (MIT, generator+detector, sample-level localization, ~16 kHz, extremely fast single-pass detector) is the one fully-open, self-hostable watermark detector. Google/DeepMind **SynthID** marks Lyria, NotebookLM audio and (via partnership) ElevenLabs and some OpenAI output, but its audio detector is gated (waitlist portal, no public local API). ElevenLabs runs a free AI Speech Classifier + SynthID-based detector for its own output. **Coverage gap: open-source TTS/voice-clone tools (the majority of malicious deepfakes) and many commercial systems do not watermark**, and watermarks degrade under speed changes and noise, confirming the maintainer's instinct that watermark-only detection is insufficient.

**4. The commercial market is real-time-fraud-focused and self-reports ~94–99% accuracy, but independent tests show a large real-world gap.** Pindrop (Pulse/Pulse Inspect), Resemble AI (DETECT-2B/3B), Reality Defender, Hive, Hiya/Loccus.ai, and Phonexia lead. Their public methodology descriptions reveal the same underlying techniques as the open-source world (SSL representations, artifact/"fakeprint" analysis, Mamba-SSM/ResNet back-ends). None are self-hostable in the way the maintainer prefers except Phonexia (Docker/Helm) and Resemble (offers on-prem).

**5. Generalization to unseen synthesis + robustness to compression/telephony is the central unsolved problem**, and it bears directly on Voxint's possibly-compressed, multilingual, real-world audio.

## Details

### A. Academic foundations, challenges, and datasets

**The ASVspoof series** (asvspoof.org) defines the field. Run by Yamagishi, Evans, Kinnunen, Todisco, Delgado, Wang, Jung et al.:
- **ASVspoof 2015**: first challenge, TTS/VC synthetic speech.
- **ASVspoof 2017**: replay attack detection.
- **ASVspoof 2019**: split into **Logical Access (LA)** (TTS/VC) and **Physical Access (PA)** (replay). ASVspoof 2019 LA remains the most common training set; SSL-AASIST reaches ~0.83% EER in-domain here, but this over-optimistically flatters real-world performance: the same XLS-R+AASIST model that scores 0.83% in-domain has been measured at **24.84% EER out-of-domain against 8 unseen TTS systems, a ~30× degradation** (arXiv:2601.02914).
- **ASVspoof 2021**: added the **Deepfake (DF)** track (compressed, no ASV, EER metric), plus LA with codec/transmission effects. The 2021 summary (54 teams) concluded detectors "lack generalization across different source datasets," the foundational warning.
- **ASVspoof 5 (2024)**: crowdsourced data, vastly more speakers, adversarial attacks for the first time, new metrics (a-DCF, min-DCF). Two tracks: stand-alone deepfake detection and spoofing-robust ASV (SASV). Winning systems used SSL + AASIST/ResNet fusion; e.g., AASIST3 (KAN-enhanced) reported min-DCF 0.1414 (open condition) and the SHADOW team's FwSE-ResNet34 hit min-DCF 0.44 (47% over baseline). Even top systems showed large error increases against the newest generators.

**Other key datasets:**
- **In-the-Wild (ITW)** (Müller et al., 2022): ~38 hours, 58 public figures, real vs. spoofed from social media and web, the de facto real-world generalization benchmark. Detectors trained on ASVspoof degrade 1–2 orders of magnitude here.
- **MLAAD (Multi-Language Audio Anti-Spoofing Dataset)** (Müller/Kawa et al., Fraunhofer AISEC + Resemble AI): the key multilingual resource. v10 = 1002.9 hours, 175 TTS models, 54 languages, built on M-AILABS. **CC-BY-NC 4.0 (non-commercial)**, HuggingFace `mueller91/MLAAD`. Best complementary training resource to ASVspoof 2019.
- **ADD 2022/2023**: Mandarin-focused; low-quality, partial-fake, and algorithm-recognition tracks.
- **Deepfake-Eval-2024** (Chandra et al., TrueMedia.org/UW, CVPR-W 2026; arXiv:2503.02857): 56.5 hrs in-the-wild 2024 audio, 52 languages, 88 sites. **The most important recent finding:** the authors observe "an average drop in AUC of 50% for video, 48% for audio, and 45% for image models when evaluated on Deepfake-Eval-2024," with off-the-shelf open-source audio models reaching a maximum AUC of only ~0.58; commercial and fine-tuned models did better but still below human forensic analysts. This is the single most decision-relevant paper for Voxint.
- Newer specialized sets: **CodecFake / Codecfake** (neural-codec artifacts), **SpoofCeleb** (real-world noisy), **SingFake / CtrSVDD** (singing voice), **MLAADv9/ML-ITW / XMAD-Bench** (multilingual in-the-wild), **PartialSpoof / LlamaPartialSpoof** (partial/segment-level fakes), **AUDETER** (open-world).

**Key recurring research findings relevant to Voxint:**
- **Compression/telephony is a top failure mode.** ASVspoof 2021 DF exists precisely because online-posted, re-compressed audio breaks detectors. Resemble's Interspeech 2025 replay study showed W2V2-AASIST EER jumping from 4.7% to 18.2% under replay; adaptive retraining with room impulse responses recovered it to ~11%.
- **Sample-rate/high-frequency reliance:** detectors often key on high-frequency synthesis artifacts, so downsampling/telephony (8 kHz) sharply raises EER.
- **Threshold transfer failure:** the 2025 audit "When EER Hides Deployment Failure" (arXiv:2606.21584) showed an SSL-AASIST model with 11.2% ITW EER produced a 39.5% HTER (driven by a 78.7% false-rejection rate on bona fide speech) when its ASVspoof threshold was transferred unchanged, meaning **calibration on your own data is essential**, not optional.
- **Explainability** is an active area (SHAP/Grad-CAM on spectrograms, prosody-aware models like HuLA, LLM-based reasoning like SARA/ALLM4ADD) but not yet production-ready; expect a score, not an explanation.

### B. Open-source projects and models (detail)

| Project | Architecture | License (code / weights) | Reported EER | Self-hosting notes |
|---|---|---|---|---|
| **NII AntiDeepfake** (`nii-yamagishilab/AntiDeepfake`) | XLS-R / wav2vec2 / MMS SSL + FC head, "post-training" on 74k hrs | **BSD-3 / CC BY-NC-SA 4.0 (non-commercial weights)** | **1.23% ITW (XLS-R-2B, zero-shot)**; 2.23% DEEP-VOICE; 4.67% ADD2023 | HF weights, 16 kHz any-length, binary output; PyTorch+fairseq. Best accuracy; **non-commercial weights** |
| **SSL_Anti-spoofing** (`TakHemlata/SSL_Anti-spoofing`) | wav2vec2-XLS-R + AASIST | **MIT / MIT** | 0.82% ASVspoof21-LA; 2.85% ASVspoof21-DF; ~10.5% ITW | 318M params, PyTorch+fairseq, Google-Drive weights; the reference baseline |
| **Nes2Net** (`Liu-Tianchi/Nes2Net_ASVspoof_ITW`) | wav2vec2-XLS-R + Nes2Net-X (~511k param back-end) | **No license file (all-rights-reserved)** | 1.49% ASVspoof21-DF; ~5.2–5.8% ITW | Very lightweight back-end; SOTA acc/param; **contact author re: license** |
| **clovaai/aasist** | SincNet + spectro-temporal GAT | MIT / MIT | 0.83% ASVspoof19-LA | Original AASIST; no SSL front-end, weak on 2021-DF/ITW |
| **DeepFense** (`deepfense.github.io`) | Modular framework (WavLM/wav2vec2/EAT/MERT × AASIST/Nes2Net/MLP/TCM) | Open (check repo) | Varies; 455+ checkpoints, 12 datasets | Best for benchmarking/experimentation on your own data |
| **AASIST3** (`lab260/AASIST3`) | KAN-enhanced AASIST + wav2vec2 | **CC BY-NC-ND 4.0 (non-commercial)** | min-DCF 0.1414 (ASVspoof5 open) | ASVspoof 2024 entry; non-commercial |
| Various HF wav2vec2 fine-tunes (e.g., `MelodyMachine/Deepfake-audio-detection-V2`) | wav2vec2-base + classifier | Mostly Apache-2.0 | 99%+ *in-domain* (misleading) | Easy `transformers` pipeline but poor generalization; several model cards explicitly warn their metrics are in-domain only |

Notes: All models expect **16 kHz mono**; AASIST-family models use ~4s (64,600-sample) windows, while AntiDeepfake accepts arbitrary length (longer = lower EER, e.g., XLS-R-1B improved from 11.86% @4s to 8.28% @50s on Deepfake-Eval-2024). RawBoost augmentation (convolutive/impulsive/stationary noise) is the standard robustness technique in this ecosystem.

### C. Watermarking (detail)

- **Meta AudioSeal** (`facebookresearch/audioseal`, **MIT incl. weights**): generator+detector, sample-level (1/16,000 s) localization, robust to compression/re-encoding/noise, single-pass detector up to ~2 orders of magnitude faster than prior work, streaming support. Detector is public and trivially self-hostable (`pip install audioseal`). But it only detects **AudioSeal-embedded** marks, used in Meta's own Audiobox/Seamless demos, not by third-party TTS. Now part of the broader `facebookresearch/content-seal` suite.
- **Google/DeepMind SynthID**: embeds inaudible marks in Lyria (music) and NotebookLM audio; extended via partnership to **ElevenLabs** and (images) **OpenAI**. Detection is via a **gated portal (waitlist)**, with **no public/local audio detector API**, so it is not integrable into Voxint. SynthID Text is open-sourced, but audio is not. Google reports 100B+ items watermarked, but only Google-ecosystem content is detectable.
- **ElevenLabs**: free **AI Speech Classifier** + SynthID-based **Audio Detector** for its own generations only.
- **Other open watermarking**: WavMark, SilentCipher, Timbre Watermarking (all proactive embed schemes; AudioSeal supersedes WavMark on robustness/speed). NII's `antispoofing-watermark` and FakeMark explore watermark+detector hybrids.
- **Coverage gap (why watermark-only fails):** open-source generators (Coqui/XTTS, Bark, Piper, F5-TTS, etc.) and many commercial ones embed nothing; watermarks are stripped by speed changes, heavy compression, or re-recording ("Watermarking Without Standards" arXiv study showed benign transforms substantially degrade AudioSeal detectability). The maintainer's plan to prioritize passive detection is correct.

### D. Commercial / SaaS landscape (reverse-engineered capabilities)

| Vendor | Claimed approach | Self-reported accuracy | Input/latency/langs | API/deploy | Notes |
|---|---|---|---|---|---|
| **Pindrop** (Pulse, Pulse Inspect) | "Liveness"/"fakeprint", a low-rank vector of spectral/temporal artifacts; trained on 350+ generation tools, 20M+ unique utterances, 40+ langs (>90% of internet's spoken languages) | 99% on file audio; >90% on unseen ("zero-day"); <1% FPR | 2–4 s segments; 8 kHz phone + full-band; language-agnostic | Cloud API/web; contact-center + Pulse Inspect for media files | Strongest independent showing (NPR: 81/84 = 96.4% vs. nearest competitor 47/84 = 56%; FTC Voice Cloning Challenge Recognition Award, large-org category, April 2024). Not self-hostable |
| **Resemble AI** (DETECT-2B / 3B-Omni / World) | Wav2Vec2 + Mamba-SSM ensemble, SSL reps, frame-by-frame | 94–98%; DETECT-3B ~98.3% on OOD test; EER <6%; DETECT-World claims 99.5% on Podonos | 200 ms latency; 30–51 langs; codec-robust (MP3/OGG/AAC, G.711/723) | API; **offers on-prem/air-gapped** + open weights on HF for DETECT model | Most Voxint-aligned commercial option; co-authors MLAAD |
| **Reality Defender** | Multi-model ensemble; partnered w/ ElevenLabs (295+ hrs synth data) | High (varies) | Audio/video/image/text; API/SDK; free tier 50 scans | Cloud; real-time | Podonos benchmark flagged very high FPR (~54%) in one test |
| **Hive AI** | Per-media classifiers; audio in 10-s chunks | Independent (U. Chicago) rated strong; DoD contract | Any language; ~$1–6/1000 calls; **on-prem option** | REST API, enterprise app | Content-moderation breadth; audio one of many models |
| **Hiya / Loccus.ai** | Deep-learning voice detection; real-time | Benchmark-leading (self-claim) | Multi-lang, live calls + files; API | Cloud API | Loccus acquired by Hiya 2024 |
| **Phonexia** | Referential deepfake detection | n/a | Docker/Helm GPU images public | **Self-hostable (Docker/Helm)** | Closest to Voxint's deployment model among vendors |
| **ElevenLabs** | Provider-specific classifier + SynthID | 99% (own content) | Free web tool | Web only | Only detects ElevenLabs output |
| **Others** | AI or Not, Sensity, DeepMedia, Deepgram (STT/TTS, not detection), McAfee, Corsound, Aurigin, Whispeak | Varies | n/a | Cloud | Deepgram is STT/TTS, **not** a deepfake detector |

**Independent benchmark reality-check (Podonos 2026, 4,524 private clips vs. modern voice clones):** Resemble AI 98.05% acc (1.4% FNR) / Pindrop 95.05% (fast: 282 ms) / Hive 83.5% (2.4% FPR but 30.5% FNR, conservative, misses many fakes) / Reality Defender 71.3% (53.7% FPR, flags more than half of real audio as fake, RTF 1.52) / **off-the-shelf Wav2Vec2 (2019 LA) only 62.9%, AST (ASVspoof 5) 56.8%**. This quantifies exactly why an un-tuned open-source model won't match a maintained commercial service on modern attacks, and why fine-tuning and refreshing on current data matters.

### E. Practical integration guidance for Voxint

Voxint's architecture (FastAPI + Celery/Redis + Docker model services behind versioned HTTP contracts, Postgres+pgvector, K3s, RTX 3090/3060 GPUs, Whisper+Pyannote+TitaNet already sharing a GPU) is an excellent fit for adding a detector as **one more containerized model service with a versioned HTTP contract**, mirroring how the TitaNet/pyannote services already work.

**Recommended design:**
1. **Per-speaker-turn detection leveraging existing Pyannote diarization.** You already produce diarized speaker turns; run the detector on each turn (or on merged same-speaker segments ≥4 s, ideally longer). This gives per-speaker synthetic scores, localizes partial fakes, and mirrors AntiDeepfake's finding that longer inputs lower EER. Aggregate to a per-file and per-speaker synthetic-likelihood.
2. **Pipeline placement:** new Celery task after diarization, before/parallel to LLM polishing. Input = audio segment or diarized turn (resampled to 16 kHz mono); output = calibrated `synthetic_probability` + `model_version` + `threshold` stored in Postgres (a new column/table keyed to speaker turn). You could optionally store detector embeddings for later re-scoring; the score itself needs no pgvector.
3. **Two-stage detection:** (a) fast **AudioSeal watermark check** (near-free, flags Meta-ecosystem content and any AudioSeal-marked audio); (b) **passive SSL detector** for everything else. Optionally add a SynthID note in docs that watermarked ElevenLabs/Google content exists but isn't locally detectable.
4. **GPU fit:** an XLS-R-2B detector is ~2B params (heavy; fits comfortably on a 3090, tight on a 3060); W2V2-AASIST (318M) or Nes2Net (~300M front-end + tiny back-end) fit easily on a 3060 alongside existing models. Batch/offline processing means you can queue detection without real-time constraints.
5. **Calibration & thresholds:** do NOT ship the paper's threshold. Calibrate on a held-out slice of *your* domain audio (the threshold-transfer research shows why). Expose the score; let the review UI flag "possibly synthetic" for human adjudication, consistent with Voxint's existing "machine proposals stay separate from your rulings" philosophy.
6. **Multilingual:** if your data is multilingual, prefer XLS-R/MMS-based models (128+/1000+ languages) and consider fine-tuning with MLAAD (non-commercial) for research use.

**Realistic accuracy expectations:** On clean, studio-quality, known-generator speech expect very high accuracy (>95%). On compressed/telephony/social-media audio with unseen 2025–2026 generators, expect the ~48% AUC drop documented in Deepfake-Eval-2024: real-world accuracy may fall to the 70–85% range and degrade further over time as new TTS ships. Plan for periodic re-training/refresh as a maintenance discipline.

## Recommendations

**Stage 1: Prototype (2–3 options, in priority order):**
1. **Tak SSL_Anti-spoofing (W2V2-AASIST), MIT**: start here. Permissive license (commercial-safe), well-documented, the community reference baseline, straightforward PyTorch. Wrap as a Docker model service with a `/detect` HTTP contract taking a 16 kHz segment.
2. **NII AntiDeepfake (XLS-R-300M or wav2vec2-large variant)**: add as the accuracy benchmark. Best real-world generalization (1.23% ITW for the 2B model). **Use only if your use is research/internal/non-commercial** (weights are CC BY-NC-SA 4.0); the BSD-3 code lets you retrain your own weights if you need commercial rights. Prefer the smaller variants for 3060 fit.
3. **Nes2Net (wav2vec2-Nes2Net-X)**: evaluate for its excellent accuracy-per-parameter if you want a lightweight back-end, **but resolve the missing-license issue with the author before any deployment.**

Run all three through **DeepFense** or a small harness against **a held-out slice of your own 100K-file corpus** (the maintainer benchmarks on real domain data, not leaderboards) plus **In-the-Wild** and **MLAAD** as external sanity checks. Compare EER, and critically **FPR at a fixed operating threshold** on your bona fide audio (false-positives create the most user friction).

**Stage 2: Add watermark pre-check:** integrate **Meta AudioSeal** (MIT, `pip install audioseal`) as a fast first pass. Document that SynthID/ElevenLabs-watermarked content is not locally detectable.

**Stage 3: Productionize the winner:** containerize behind a versioned HTTP contract like your TitaNet service; add per-speaker-turn scoring; store `synthetic_probability`, `model_version`, `threshold` in Postgres; expose "possibly synthetic" flags in the review UI (never auto-decisions); pin the model and hold it to measured-equivalence gates like your other models.

**Stage 4: Maintenance discipline:** schedule periodic re-evaluation/fine-tuning as new TTS systems appear. Keep a small labeled set of recent real+fake domain audio for continuous calibration.

**Thresholds that would change the recommendation:**
- If you need **commercial-grade real-world accuracy today with minimal ML effort** and can tolerate a cloud dependency, then **Resemble AI DETECT (offers on-prem + open HF weights)** or **Phonexia (Docker/Helm)** are the only vendors matching your self-host preference; otherwise Pindrop for pure accuracy.
- If FPR on your bona fide audio exceeds ~5–10% at a usable threshold after calibration, invest in fine-tuning on domain data before deploying, or fall back to flag-for-review-only.
- If your audio is heavily telephony/8 kHz, prioritize models with codec augmentation (retrain with RawBoost + codec/RIR augmentation) and expect a meaningful accuracy penalty.

## Caveats
- **Vendor accuracy claims (94–99%) are self-reported on proprietary test sets** and are not comparable; independent benchmarks (Deepfake-Eval-2024, Podonos) show much lower real-world numbers. Treat all marketing figures skeptically.
- **EER is an optimistic, oracle-threshold metric.** Deployed performance depends on a fixed threshold and is often far worse (the HTER/threshold-transfer finding). Always calibrate and monitor FPR/FNR separately.
- **Generalization to unseen generators is the fundamental unsolved problem.** Any model you deploy is decaying against new TTS from the day you ship it (recall the 30× in-domain-to-out-of-domain degradation for XLS-R+AASIST).
- **License caveats are load-bearing:** AntiDeepfake weights, MLAAD and AASIST3 are non-commercial; Nes2Net has no license; SSL_Anti-spoofing, clovaai/aasist and AudioSeal are permissive (MIT). Confirm your intended use (personal/research vs. commercial) before selecting.
- **SynthID audio detection is not locally available** (gated portal), so it cannot be integrated into a self-hosted pipeline as of this writing.
- Some 2026-dated sources in this report are vendor blogs and secondary aggregators; where possible primary papers (arXiv, ISCA, HuggingFace model cards, GitHub LICENSE files) were used for technical claims.
