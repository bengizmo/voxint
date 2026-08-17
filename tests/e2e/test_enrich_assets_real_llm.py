"""Real-LLM E2E: drive **summary** generation through the real ``HttpLLMClient``
against a real OpenAI-compatible endpoint, and gate the *chain*, not the prose.

Every other test in the suite feeds enrichment a ``FakeLLM``/``FailingLLM``; this
is the one lane that proves the real adapter → real endpoint → durable job → asset
path actually closes. What it asserts is the plumbing and the persistence
contract:

1. the endpoint is reachable through the real adapter (the ``llm_config`` gate),
2. the durable job reaches ``succeeded`` with ``error`` NULL,
3. a *current* summary asset persists with the expected kind, producer/prompt
   version, model alias, ``config`` snapshot, and a well-formed ``source_content_hash``,
4. that asset is *non-stale* immediately after generation,
5. one real operator correction re-stales it (``summary`` reappears in
   ``kinds_needing_generation``),
6. malformed/unusable model output yields an honest ``failed`` job — no asset,
   no partial success.

What it deliberately does NOT assert: the summary's semantic quality (length,
topics, wording, grounding). Those shape invariants are enforced deterministically
in ``enrichment/run_assets_llm.py`` and covered by ``tests/integration``; here a
real, temperature-0-but-nondeterministic local model produces the text, so any
assertion on it would be a flake, not a gate. The summary is *characterized*
(printed) for the operator, never blocked on.

Seeding choice (stated honestly): the transcript is seeded, not produced by the
real pipeline. Phase 1 (``test_real_pipeline.py``) already gates ASR + diarization
+ embedding; feeding this lane a fixed transcript isolates the LLM boundary, so a
failure here names the LLM chain rather than an upstream model service, and keeps
the LLM's *input* deterministic (the endpoint stays the only moving part).
"""

from __future__ import annotations

import json
import re
import uuid

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.adjudication.review_state import set_correction
from voxint.app_settings import complete_onboarding
from voxint.clients.llm import HttpLLMClient
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    PipelineRun,
    RunAssetJob,
    RunAssetJobStatus,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
    TranscriptSegment,
)
from voxint.enrichment.asset_jobs import create_jobs, execute_job, kinds_needing_generation
from voxint.enrichment.producers.run_assets_llm import (
    PRODUCER_NAME,
    PRODUCER_VERSION,
    PROMPT_VERSION,
)
from voxint.enrichment.run_assets import latest_assets, load_source, source_content_hash

from .conftest import LLMConfig

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

# A short, self-contained multi-speaker transcript with named people/orgs so the
# summary has real material to abstract. Deterministic — it is the LLM's input,
# and the endpoint is the only nondeterministic element in the chain.
_TRANSCRIPT: tuple[tuple[str, str], ...] = (
    (
        "S0",
        "Hi everyone, I'm Joanne from Acme Corp, and today we're reviewing "
        "the Q3 widget rollout.",
    ),
    (
        "S1",
        "Thanks Joanne. The rollout hit 82 percent of target regions, but the "
        "Denver warehouse had a delay.",
    ),
    (
        "S2",
        "From logistics, the Denver delay was a carrier issue; we've switched "
        "carriers and expect recovery next week.",
    ),
    (
        "S0",
        "Good. So overall we're on track, with Denver as the one open risk. "
        "Let's reconvene Friday.",
    ),
)


def seed_completed_run(session_factory: sessionmaker[Session]) -> uuid.UUID:
    """Persist an onboarded, COMPLETED run with the fixed transcript.

    Onboards the app row with ``llm_enabled=True`` so the enrichment gate — which
    resolves row-over-env (issue #10) — is open; an onboarded row left at the
    default ``False`` would close it even with env ``LLM_ENABLED`` set.
    """
    with session_factory() as session:
        complete_onboarding(session, llm_enabled_default=True)
        media = MediaItem(source_path=f"e2e/{uuid.uuid4().hex}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        for index, (label, text) in enumerate(_TRANSCRIPT):
            session.add(
                TranscriptSegment(
                    pipeline_run_id=run.id,
                    segment_index=index,
                    start_seconds=float(index * 5),
                    end_seconds=float(index * 5 + 5),
                    raw_text=text,
                    diarization_label=label,
                )
            )
        run_id = run.id
        session.commit()
    return run_id


def _create_summary_job(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, settings: Settings
) -> uuid.UUID:
    with session_factory() as session:
        created, _already = create_jobs(
            session, pipeline_run_id=run_id, kinds=(RunAssetKind.SUMMARY,), settings=settings
        )
        assert len(created) == 1, "expected exactly one summary job"
        session.commit()
        return created[0].id


def test_real_llm_summary_chain(
    session_factory: sessionmaker[Session],
    settings: Settings,
    llm_config: LLMConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Steps 1-5: the real summary chain succeeds, persists correct metadata,
    is non-stale, and re-stales on a real operator correction."""
    run_id = seed_completed_run(session_factory)
    job_id = _create_summary_job(session_factory, run_id, settings)

    # Step 1-2: the REAL client (llm=None → execute_job builds HttpLLMClient from
    # the frozen job snapshot) drives one generation to a durable, honest verdict.
    execute_job(session_factory, job_id, settings=settings, llm=None)

    with session_factory() as session:
        job = session.get(RunAssetJob, job_id)
        assert job is not None
        assert job.status == RunAssetJobStatus.SUCCEEDED.value, (
            f"summary job did not succeed against {llm_config.resolved_identity}: "
            f"status={job.status} error={job.error!r}"
        )
        assert job.error is None
        assert job.asset_id is not None

        # Step 3: a current summary asset with the expected provenance + snapshot.
        asset = session.get(RunEnrichmentAsset, job.asset_id)
        assert asset is not None
        assert asset.asset_kind == RunAssetKind.SUMMARY.value
        assert asset.superseded_by_asset_id is None, "freshly generated asset is not current"
        assert asset.producer == PRODUCER_NAME
        assert asset.producer_version == PRODUCER_VERSION
        assert asset.model == settings.llm_model, "asset model alias != configured LLM_MODEL"
        assert _SHA256_HEX.match(asset.source_content_hash or ""), (
            f"source_content_hash is not a sha256 hex: {asset.source_content_hash!r}"
        )
        assert isinstance(asset.config, dict)
        assert asset.config["prompt_version"] == PROMPT_VERSION
        assert asset.config["producer_version"] == PRODUCER_VERSION
        assert asset.config["model"] == settings.llm_model

        # The stored hash must equal a fresh recompute over the same source, or
        # the asset would be born stale.
        assert asset.source_content_hash == source_content_hash(load_source(session, run_id))

        # Step 4: summary is NOT among the kinds still needing generation (the
        # other two kinds legitimately are — they were never generated).
        needing = kinds_needing_generation(session, run_id)
        assert RunAssetKind.SUMMARY not in needing, f"summary reported stale immediately: {needing}"

        summary_text = str(asset.payload.get("summary", ""))

    # Characterization only — NEVER a blocking assertion (a real model's prose is
    # not a contract). Reported so the operator can eyeball quality + provenance.
    with capsys.disabled():
        print(
            f"\n[characterization] resolved_identity={llm_config.resolved_identity} "
            f"model={llm_config.model}\n[characterization] summary "
            f"({len(summary_text)} chars): {summary_text}"
        )

    # Step 5: one real operator correction changes the source text → the summary
    # asset becomes stale (summary reappears in kinds_needing_generation). This is
    # the review route's staleness contract (issue #58) exercised end to end.
    with session_factory() as session:
        segment = (
            session.query(TranscriptSegment)
            .filter_by(pipeline_run_id=run_id, segment_index=0)
            .one()
        )
        set_correction(
            session,
            segment=segment,
            text="Hi everyone, I'm Joanne from Beta Industries reviewing the Q4 gadget launch.",
        )
        session.commit()
        needing_after = kinds_needing_generation(session, run_id)
        assert RunAssetKind.SUMMARY in needing_after, (
            f"operator correction did not re-stale the summary: {needing_after}"
        )


class _MockLLMClient:
    """A real ``HttpLLMClient`` over a canned in-memory transport.

    Honesty boundary for step 6: a real model cannot be *coerced* into emitting
    deterministic garbage, so this drives the REAL adapter (envelope parsing),
    the REAL producer (``_parse_summary``), the REAL job finalization, and the
    REAL database against a controlled 200 reply carrying an unusable body. Only
    the endpoint bytes are canned; every code path that turns them into a FAILED
    verdict is production code.
    """

    def __init__(self, body: dict[str, object]) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(body)}}]}
            )

        self._transport = httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://stub.invalid"
        )
        self.client = HttpLLMClient(
            base_url="http://stub.invalid",
            model="stub",
            api_key="",
            timeout_seconds=30.0,
            client=self._transport,
        )

    def __enter__(self) -> HttpLLMClient:
        return self.client

    def __exit__(self, *exc: object) -> None:
        self._transport.close()


def test_malformed_summary_reply_is_honest_failure(
    session_factory: sessionmaker[Session],
    settings: Settings,
    llm_config: LLMConfig,
) -> None:
    """Step 6: a 200 reply with an unusable summary fails the job honestly —
    ``failed`` status, a closed-vocabulary error, and NO asset (not partial)."""
    run_id = seed_completed_run(session_factory)
    job_id = _create_summary_job(session_factory, run_id, settings)

    # A well-formed envelope whose summary is blank → _parse_summary rejects it.
    with _MockLLMClient({"summary": ""}) as llm:
        execute_job(session_factory, job_id, settings=settings, llm=llm)

    with session_factory() as session:
        job = session.get(RunAssetJob, job_id)
        assert job is not None
        assert job.status == RunAssetJobStatus.FAILED.value, f"expected FAILED, got {job.status}"
        assert job.error is not None and "no usable summary" in job.error, (
            f"error should name the unusable summary, got {job.error!r}"
        )
        assert job.asset_id is None, "a failed job must record no asset"
        assert "summary" not in latest_assets(session, run_id), (
            "a failed generation must not leave a current summary asset"
        )
