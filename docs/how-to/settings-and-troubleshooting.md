# Settings and troubleshooting

*A how-to guide for Voxint operators (v0.17.0).*

Almost everything about Voxint is configurable from your browser. Open the
console at `http://127.0.0.1:8080/`, sign in (it uses a simple username and
password), and click **Settings** in the top navigation bar. Every option on
this page saves right there — no files to edit, no services to restart.

Changes you save do **not** touch runs that are already in progress. They apply
to your **next** run or job. That is by design: a run's settings are locked in
the moment it starts, so nothing shifts underneath a job that is already going.

![The Voxint Settings page, showing the First-run setup, Features, Media
folders, LLM enhancement, Sources and research, and Guided tutorial
sections stacked down the page.](../images/settings.png)

---

## The Settings page, section by section

The page is a stack of self-contained sections. Each one has its own **Save**
button and saves on its own — saving one section never disturbs another. Here is
what each does.

### First-run setup

The setup wizard that ran the first time you opened Voxint is always available
again. Click **Re-run the setup wizard** to walk through it any time.

Re-running is safe: the wizard **never resets your existing preferences** unless
you deliberately change something as you go through it. Use it when you want to
revisit the guided sequence — for example to re-check your media folder or the
service readiness screen.

For a full walkthrough of the wizard, see the [onboarding
guide](../onboarding.md).

### Features

This section turns Voxint's **optional** features on and off. Each feature has
three settings:

- **On** — always use it.
- **Off** — never use it.
- **Use installation setting** — follow whatever was configured when Voxint was
  installed. Choosing this is how you *undo* an override: it hands the decision
  back to the installation default, so you are never permanently pinned to a
  choice you made here. The label tells you what that default currently is
  (On or Off).

The optional features are:

- **Speaker name suggestions** — scans finished transcripts for likely speaker
  names. This runs fully on your machine and needs no LLM.
- **LLM name pass** — additionally asks the enhancement LLM to propose speaker
  names. This one needs both LLM enhancement (below) and Speaker name
  suggestions to be on.
- **Run assets (summary, topics, entities)** — generates a summary, a topic
  list, and grounded entity mentions for each run. Needs LLM enhancement.
- **Auto-generate run assets** — starts that run-asset generation automatically
  the moment a run is finalized, instead of you asking for it. Needs Run assets
  to be on.
- **Download media from a URL** — lets you submit media by pasting a URL, which
  Voxint fetches with yt-dlp. This is independent of the LLM features. Note that
  fetching a URL reaches out to the network (see the last troubleshooting entry).

If LLM enhancement is off, the page reminds you that the LLM-dependent features
above stay inactive even if you switch them On here — turn on LLM enhancement
first.

### Media folders

Voxint watches folders **inside your media root** and processes what you put in
them. In this section you register those watched folders and pick a **domain
pack** for each one.

A domain pack tunes transcription and enrichment for a particular kind of
recording (for example, a pack with the right vocabulary and prompts for your
subject area). Assigning packs per folder lets you keep, say, interviews and
lectures tuned differently. Browsing stays inside your media root — you cannot
wander off into the rest of the disk. Changes apply to the next job you submit.

To learn what domain packs are and how to choose one, see
[domain-packs.md](../domain-packs.md).

### LLM enhancement *(optional)*

When enabled, Voxint sends transcript segments to an OpenAI-compatible model to
clean them up and suggest likely speaker names. This is **best-effort**: a slow
or failing model never blocks a run — the run still completes, just without the
enhancement.

You configure three things:

- **Endpoint base URL** — the address of your OpenAI-compatible model, for
  example `https://llm.example.com/v1`. Leave it blank to use the installation
  setting.
- **Model** — the model name to request. Blank uses the installation setting.
- **API key** — the credential for that endpoint. It is stored in Voxint's own
  local database and used for all LLM features. For your safety it is **never
  shown again after you save it**; leave the field blank to keep the key you
  already saved.

A key you save here **wins over** a key set in the environment. If you saved one
and want to fall back to the environment key, tick **Remove saved key**.

Before enhancement can be enabled you need a key configured (either saved here or
in the environment), and the LLM time budget has to fit within the transcription
stage's lease — the page tells you if it does not.

### Sources and research *(optional, off by default)*

Voxint can research the people in your recordings **on the web** to suggest
names and background. This is off unless you turn it on, and it is the one part
of Voxint that deliberately reaches out to the internet.

- **Web research** — the master switch. When **On**, Voxint makes outbound web
  requests to the search provider you configure to research speakers. When
  **Off**, Voxint never touches the network for research. Like the Features
  toggles, it also offers **Use installation setting**.
- **Web-research enrichment producer** — generates researched background for
  each run. It needs both Web research (above) and LLM enhancement to be on.
- **Search provider endpoint** — the address of the search instance Voxint
  queries. Enter just the endpoint address, with no query string, fragment, or
  embedded credentials.
- **Search provider API key** — only needed if your search provider requires
  one. Stored locally in Voxint's database and, like the LLM key, never shown
  again after saving.
- **Trusted domains** — an optional list of domains you trust. Drafts that cite
  these domains are given more confidence during review. This does **not** block
  anything: other domains are still used and researched normally. Enter bare
  domains separated by commas, spaces, or new lines.

If LLM enhancement is off, this section reminds you that web-research enrichment
stays inactive until you enable it.

### Guided tutorial

A short, hands-on walkthrough of one full run on a bundled three-speaker sample —
submit, review, attribute, export — with no command line needed.

- If you have not set it up yet, **Set up and start the guided tutorial** stages
  the sample and drops you straight into it.
- Once you have set it up, you can **Start** it.
- After you finish, the button becomes **Replay**. Replaying walks you through
  the sample again and is **non-destructive**: your previous speaker rulings on
  the tutorial run are preserved, not reset.

---

## Troubleshooting

Start with the [onboarding troubleshooting
list](../onboarding.md#troubleshooting) — it covers the most common
setup-time problems. The short version, plus a few review-console questions, is
below.

### Setup and settings

- **The console keeps sending me back to `/setup`.** This is expected until
  onboarding is finished. The gate holds every page there until you complete the
  wizard. Finish it and you are released.
- **I changed a vocabulary or LLM setting and it "did nothing."** Settings are
  snapshotted when a run starts. Your change applies to the **next** submission,
  not to a run already in flight. Submit again to see it take effect.
- **The setup wizard's service check shows everything down.** The core stack does
  not include the model services; you start those with an overlay (a GPU or CPU
  overlay, depending on your machine). The check is advisory — you can finish
  setup either way — but see the next entry for what happens if you submit a run
  while a service is down.

![The setup wizard's readiness-check screen, listing each Voxint service with a
status indicator so you can see at a glance what is up and what is
down.](../images/setup-wizard.png)

- **Enhancement won't turn on.** Either no API key is configured (neither a saved
  key nor one in the environment), or the LLM time budget does not fit the
  transcription stage's lease. The LLM section tells you which. Enter a key to
  fix the first; the second is an installation setting your maintainer can
  adjust.

### The review console

- **I submitted a run while a model service was down, and it failed.** A run
  submitted while a service it needs is unavailable does not fail instantly — it
  retries with backoff (roughly five attempts over about an hour and a half) and
  only then lands as **failed**. Bring the services up, then **requeue** the run
  from its page. Nothing is lost; it simply starts over cleanly.
- **The transcript shows fewer (or more) speakers than I heard.** This is usually
  correct behavior being misread rather than a bug — for example two people with
  similar voices merged, or one person split across a noisy passage. Before
  correcting it by hand, read
  [interpreting-diarization.md](../interpreting-diarization.md), which explains
  what the diarizer is doing and how to read its output.
- **"Does nothing leave my machine?"** By default, all processing — transcription,
  diarization, speaker identification, and review — happens **locally** on your
  hardware, and no audio or transcript is sent anywhere. There are exactly three
  features that reach the network, and **each is off unless you, the operator,
  turn it on**: fetching media from a URL (Features), web research (Sources and
  research), and sending transcript segments to a remote LLM endpoint (LLM
  enhancement, if you point it at a remote model). Leave those off and Voxint
  stays entirely on your machine.

---

## Related guides

- [Add media & manage runs](add-media-and-manage-runs.md)
- [Review & adjudicate](reviewing-and-adjudicating.md)
- [Manage speakers & export](managing-speakers-and-exporting.md)
- [Setup](../setup.md) — install Voxint on your OS and hardware.

- [Onboarding](../onboarding.md) — the guided installer, the setup wizard, and
  the built-in tutorial.
- [Domain packs](../domain-packs.md) — what the per-folder packs do and how to
  pick one.
- [Interpreting diarization](../interpreting-diarization.md) — how to read the
  speaker output before you correct it.
