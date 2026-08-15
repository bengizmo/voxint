"""Pure logic for run-level assets (#41): source hashing, payload validation,
prompt rendering/truncation, and the producer's fail-closed reply parsing —
no DB, no HTTP.
"""

import uuid

import pytest

from voxint.config import Settings
from voxint.db.models import RunAssetKind
from voxint.enrichment.producers.run_assets_llm import (
    RunAssetProducerError,
    _locate_quote,
    _parse_mentions,
    _parse_summary,
    _parse_topics,
    build_messages,
    config_snapshot,
    render_source,
)
from voxint.enrichment.run_assets import (
    MAX_SUMMARY_CHARS,
    RunAssetError,
    RunAssetSource,
    SegmentSource,
    source_content_hash,
    validate_payload,
)

RUN_ID = uuid.UUID("00000000-0000-0000-0000-000000000041")


def make_source(
    *,
    segments: tuple[SegmentSource, ...] | None = None,
    metadata: dict[str, object] | None = None,
    operator_notes: str | None = None,
) -> RunAssetSource:
    return RunAssetSource(
        pipeline_run_id=RUN_ID,
        segments=segments
        or (
            SegmentSource(0, "S0", "Hello, I am Joanne from Acme Corp."),
            SegmentSource(1, "S1", "Thanks Ann. Let's talk about widgets."),
        ),
        metadata=metadata,
        operator_notes=operator_notes,
    )


class TestSourceContentHash:
    def test_stable_for_identical_source(self) -> None:
        assert source_content_hash(make_source()) == source_content_hash(make_source())

    def test_changes_with_text_metadata_and_notes(self) -> None:
        base = source_content_hash(make_source())
        changed_text = make_source(segments=(SegmentSource(0, "S0", "Different words entirely."),))
        with_meta = make_source(metadata={"title": "T"})
        with_notes = make_source(operator_notes="context")
        hashes = {
            base,
            source_content_hash(changed_text),
            source_content_hash(with_meta),
            source_content_hash(with_notes),
        }
        assert len(hashes) == 4

    def test_is_lowercase_hex_sha256(self) -> None:
        value = source_content_hash(make_source())
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")


class TestValidatePayload:
    def test_summary_shape(self) -> None:
        source = make_source()
        validate_payload(RunAssetKind.SUMMARY, {"summary": "A short abstract."}, source=source)
        with pytest.raises(RunAssetError, match="non-empty"):
            validate_payload(RunAssetKind.SUMMARY, {"summary": "  "}, source=source)
        with pytest.raises(RunAssetError, match="unknown keys"):
            validate_payload(RunAssetKind.SUMMARY, {"summary": "x", "extra": 1}, source=source)
        with pytest.raises(RunAssetError, match="missing keys"):
            validate_payload(RunAssetKind.SUMMARY, {}, source=source)
        with pytest.raises(RunAssetError, match=str(MAX_SUMMARY_CHARS)):
            validate_payload(
                RunAssetKind.SUMMARY,
                {"summary": "x" * (MAX_SUMMARY_CHARS + 1)},
                source=source,
            )

    def test_topics_shape(self) -> None:
        source = make_source()
        good = {
            "topics": [
                {
                    "label": "Widgets",
                    "description": None,
                    "confidence": 0.8,
                    "vocabulary": None,
                    "term_id": None,
                }
            ]
        }
        validate_payload(RunAssetKind.TOPICS, good, source=source)
        with pytest.raises(RunAssetError, match=r"1\.\.10"):
            validate_payload(RunAssetKind.TOPICS, {"topics": []}, source=source)
        with pytest.raises(RunAssetError, match="duplicate topic"):
            validate_payload(
                RunAssetKind.TOPICS,
                {"topics": [{"label": "Widgets"}, {"label": "widgets"}]},
                source=source,
            )
        with pytest.raises(RunAssetError, match="confidence"):
            validate_payload(
                RunAssetKind.TOPICS,
                {"topics": [{"label": "Widgets", "confidence": 1.5}]},
                source=source,
            )
        # v1 reserves the domain-pack fields as nulls.
        with pytest.raises(RunAssetError, match="payload v1"):
            validate_payload(
                RunAssetKind.TOPICS,
                {"topics": [{"label": "Widgets", "term_id": "t1"}]},
                source=source,
            )

    def test_mentions_shape_and_grounding(self) -> None:
        source = make_source()
        good = {
            "mentions": [
                {
                    "surface": "Acme Corp",
                    "kind": "organization",
                    "occurrences": [
                        {
                            "segment_index": 0,
                            "quote": "Acme Corp",
                            "start_char": 24,
                            "end_char": 33,
                        }
                    ],
                }
            ],
            "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
        }
        validate_payload(RunAssetKind.ENTITY_MENTIONS, good, source=source)
        # Empty mentions are a valid answer (candidates, not a verdict).
        validate_payload(
            RunAssetKind.ENTITY_MENTIONS,
            {
                "mentions": [],
                "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
            },
            source=source,
        )
        bad_quote = {
            "mentions": [
                {
                    "surface": "Acme Corp",
                    "kind": None,
                    "occurrences": [
                        {
                            "segment_index": 0,
                            "quote": "Evil Corp",
                            "start_char": 24,
                            "end_char": 33,
                        }
                    ],
                }
            ],
            "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
        }
        with pytest.raises(RunAssetError, match="does not match"):
            validate_payload(RunAssetKind.ENTITY_MENTIONS, bad_quote, source=source)
        out_of_run = {
            "mentions": [
                {
                    "surface": "x",
                    "occurrences": [
                        {"segment_index": 99, "quote": "x", "start_char": 0, "end_char": 1}
                    ],
                }
            ],
            "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
        }
        with pytest.raises(RunAssetError, match="outside the run"):
            validate_payload(RunAssetKind.ENTITY_MENTIONS, out_of_run, source=source)


class TestRenderSource:
    def test_includes_metadata_notes_and_labeled_segments(self) -> None:
        document, truncated = render_source(
            make_source(metadata={"title": "T"}, operator_notes="note"), max_chars=10_000
        )
        assert not truncated
        assert '"title": "T"' in document
        assert "note" in document
        assert "[0] S0: Hello, I am Joanne from Acme Corp." in document

    def test_truncates_head_and_tail_over_budget(self) -> None:
        long_segments = tuple(
            SegmentSource(i, "S0", f"segment {i} " + "words " * 50) for i in range(50)
        )
        document, truncated = render_source(make_source(segments=long_segments), max_chars=2_000)
        assert truncated
        assert len(document) <= 2_000
        assert "truncated for length" in document
        assert document.startswith("Transcript:")  # head survives
        assert "segment 49" in document  # tail survives


class TestLocateQuote:
    TEXT = "Hello, I am Joanne from Acme Corp."

    def test_exact_match_offsets(self) -> None:
        assert _locate_quote("Acme Corp", self.TEXT) == (24, 33)

    def test_case_insensitive_fallback(self) -> None:
        located = _locate_quote("acme corp", self.TEXT)
        assert located == (24, 33)

    def test_word_boundary_rejects_substring(self) -> None:
        # "Ann" must not anchor inside "Joanne" (the #38 lesson).
        assert _locate_quote("Ann", self.TEXT) is None

    def test_missing_quote(self) -> None:
        assert _locate_quote("Umbrella Inc", self.TEXT) is None


class TestParsers:
    def test_parse_summary(self) -> None:
        assert _parse_summary({"summary": " ok "}) == {"summary": "ok"}
        with pytest.raises(RunAssetProducerError):
            _parse_summary({"summary": ""})
        with pytest.raises(RunAssetProducerError):
            _parse_summary({})

    def test_parse_topics_dedupes_and_bounds(self) -> None:
        parsed = _parse_topics(
            {
                "topics": [
                    {"label": " Supply  chain ", "confidence": 0.9},
                    {"label": "supply chain"},  # case-insensitive duplicate
                    {"label": "Widgets", "confidence": 7},  # bad confidence → null
                ]
            }
        )
        labels = [t["label"] for t in parsed["topics"]]
        assert labels == ["Supply chain", "Widgets"]
        assert parsed["topics"][0]["confidence"] == 0.9
        assert parsed["topics"][1]["confidence"] is None
        assert all(t["vocabulary"] is None and t["term_id"] is None for t in parsed["topics"])

    def test_parse_topics_empty_is_failure(self) -> None:
        # An empty topic list is a refusal to answer, not an asset.
        with pytest.raises(RunAssetProducerError, match="no usable topics"):
            _parse_topics({"topics": []})

    def test_parse_mentions_grounds_and_reslices(self) -> None:
        source = make_source()
        parsed = _parse_mentions(
            {
                "mentions": [
                    {
                        "surface": "Acme Corp",
                        "kind": "organization",
                        # Case differs — the recorded quote is re-sliced verbatim.
                        "occurrences": [{"segment_index": 0, "quote": "acme corp"}],
                    },
                    {
                        "surface": "Ann",
                        "kind": "person",
                        # Substring of "Joanne" — dropped, not grounded.
                        "occurrences": [{"segment_index": 0, "quote": "Ann"}],
                    },
                    {
                        "surface": "Ghost",
                        "kind": None,
                        "occurrences": [{"segment_index": 99, "quote": "Ghost"}],
                    },
                ]
            },
            source,
        )
        assert [m["surface"] for m in parsed["mentions"]] == ["Acme Corp"]
        occurrence = parsed["mentions"][0]["occurrences"][0]
        assert occurrence["quote"] == "Acme Corp"  # original casing from the text
        assert occurrence["start_char"] == 24
        assert parsed["diagnostics"] == {
            "dropped_unlocatable": 1,
            "dropped_out_of_run": 1,
        }
        # The parsed payload passes the writer's validator as-is.
        validate_payload(RunAssetKind.ENTITY_MENTIONS, parsed, source=source)

    def test_parse_mentions_all_dropped_is_failure(self) -> None:
        # Everything the model offered failed grounding → a failed generation,
        # never an authoritative "no entities".
        with pytest.raises(RunAssetProducerError, match="survived grounding"):
            _parse_mentions(
                {
                    "mentions": [
                        {
                            "surface": "Ghost",
                            "occurrences": [{"segment_index": 0, "quote": "Ghost"}],
                        }
                    ]
                },
                make_source(),
            )

    def test_parse_mentions_empty_reply_is_valid(self) -> None:
        parsed = _parse_mentions({"mentions": []}, make_source())
        assert parsed["mentions"] == []


class TestBuildMessagesAndConfig:
    def test_build_messages_carries_instructions_and_document(self) -> None:
        settings = Settings(_env_file=None)
        messages, truncated = build_messages(
            RunAssetKind.SUMMARY, make_source(), max_chars=settings.run_assets_max_input_chars
        )
        assert not truncated
        assert messages[0].role == "system"
        assert "summary" in messages[1].content
        assert "[0] S0:" in messages[1].content

    def test_config_snapshot_records_shape(self) -> None:
        settings = Settings(_env_file=None)
        snapshot = config_snapshot(settings, truncated=True)
        assert snapshot["model"] == settings.llm_model
        assert snapshot["truncated"] is True
        assert snapshot["max_input_chars"] == settings.run_assets_max_input_chars


class TestValidatorEdges:
    """The remaining fail-closed branches, exercised one by one."""

    def _mentions(self, **occurrence_overrides: object) -> dict[str, object]:
        occurrence: dict[str, object] = {
            "segment_index": 0,
            "quote": "Acme Corp",
            "start_char": 24,
            "end_char": 33,
            **occurrence_overrides,
        }
        return {
            "mentions": [
                {"surface": "Acme Corp", "kind": None, "occurrences": [occurrence]}
            ],
            "diagnostics": {"dropped_unlocatable": 0, "dropped_out_of_run": 0},
        }

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({"topics": "not-a-list"}, r"1\.\.10"),
            ({"topics": ["bare-string"]}, "must be an object"),
            ({"topics": [{"label": 7}]}, "non-empty string"),
            ({"topics": [{"label": "x" * 200}]}, "label over"),
            ({"topics": [{"label": "T", "description": ""}]}, "description"),
            ({"topics": [{"label": "T", "confidence": "high"}]}, "null or a number"),
            ({"topics": [{"label": "T", "vocabulary": {"id": "v"}}]}, "payload v1"),
        ],
    )
    def test_topics_rejections(self, payload: dict[str, object], message: str) -> None:
        with pytest.raises(RunAssetError, match=message):
            validate_payload(RunAssetKind.TOPICS, payload, source=make_source())

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"segment_index": "0"}, "must be an integer"),
            ({"quote": ""}, "non-empty string"),
            ({"start_char": -1}, "non-negative integer"),
            ({"end_char": 10_000}, "outside segment"),
            ({"start_char": 33, "end_char": 33}, "outside segment"),
        ],
    )
    def test_occurrence_rejections(
        self, mutation: dict[str, object], message: str
    ) -> None:
        with pytest.raises(RunAssetError, match=message):
            validate_payload(
                RunAssetKind.ENTITY_MENTIONS, self._mentions(**mutation), source=make_source()
            )

    def test_mention_shape_rejections(self) -> None:
        source = make_source()
        diagnostics = {"dropped_unlocatable": 0, "dropped_out_of_run": 0}
        with pytest.raises(RunAssetError, match="must be an object"):
            validate_payload(
                RunAssetKind.ENTITY_MENTIONS,
                {"mentions": ["bare"], "diagnostics": diagnostics},
                source=source,
            )
        with pytest.raises(RunAssetError, match="mention kind"):
            validate_payload(
                RunAssetKind.ENTITY_MENTIONS,
                {
                    "mentions": [
                        {"surface": "x", "kind": "planet", "occurrences": []}
                    ],
                    "diagnostics": diagnostics,
                },
                source=source,
            )
        with pytest.raises(RunAssetError, match="non-negative integer"):
            validate_payload(
                RunAssetKind.ENTITY_MENTIONS,
                {
                    "mentions": [],
                    "diagnostics": {"dropped_unlocatable": -1, "dropped_out_of_run": 0},
                },
                source=source,
            )
        with pytest.raises(RunAssetError, match="diagnostics must be an object"):
            validate_payload(
                RunAssetKind.ENTITY_MENTIONS,
                {"mentions": [], "diagnostics": 3},
                source=source,
            )


class TestRecordValidation:
    """_validate_record via record_asset's up-front checks (no DB is reached —
    every case fails before the first query)."""

    def _record(self, **overrides: object) -> None:
        from datetime import UTC, datetime

        from voxint.enrichment.run_assets import record_asset

        kwargs: dict[str, object] = {
            "source": make_source(),
            "kind": RunAssetKind.SUMMARY,
            "payload": {"summary": "ok"},
            "payload_schema_version": 1,
            "producer": "run_assets.llm",
            "producer_version": "1",
            "model": "m",
            "idempotency_key": "k",
            "started_at": datetime.now(tz=UTC),
            "completed_at": datetime.now(tz=UTC),
        }
        kwargs.update(overrides)
        record_asset(None, **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"producer": " "}, "producer empty"),
            ({"model": "m" * 300}, "model empty or over"),
            ({"payload_schema_version": 0}, "payload_schema_version"),
            ({"idempotency_key": "  "}, "idempotency_key"),
            ({"config": {"a": 1}}, "set together"),
            ({"config": {"a": 1}, "config_schema_version": 0}, ">= 1"),
            (
                {"config": {"a": object()}, "config_schema_version": 1},
                "not JSON-serializable",
            ),
            (
                {"config": {"a": "x" * 20_000}, "config_schema_version": 1},
                "config over",
            ),
        ],
    )
    def test_rejections(self, overrides: dict[str, object], message: str) -> None:
        with pytest.raises(RunAssetError, match=message):
            self._record(**overrides)

    def test_naive_timestamp_and_ordering(self) -> None:
        from datetime import UTC, datetime, timedelta

        with pytest.raises(RunAssetError, match="timezone-aware"):
            self._record(started_at=datetime.now())
        now = datetime.now(tz=UTC)
        with pytest.raises(RunAssetError, match="precedes"):
            self._record(started_at=now, completed_at=now - timedelta(seconds=1))
