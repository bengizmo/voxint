"""Pinned pre-refactor filter output. Regenerate only via this module's CLI."""

import argparse
import importlib
import itertools
import json
import random
import re
import uuid
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from tests.unit.filter_inputs import seeded_filler_soups
from voxint.adjudication.turns import (
    PieceRule,
    SpeakerTurn,
    TextMapping,
    TurnPiece,
    WordAnchor,
    join_pieces,
)
from voxint.export import to_markdown_turns, to_txt_turns
from voxint.export.filler_lists import (
    DEFAULT_FILLER_LIST,
    TIER_1,
    TIER_2,
    FillerList,
    effective_filler_list,
)
from voxint.export.fillers import drop_fillers_with_seams
from voxint.export.reading import layout_turns
from voxint.export.turn_filters import apply_turn_filters

_FIXTURE = Path(__file__).with_name("fixtures") / "filter_characterization.json"
_OPTIONS = (None, DEFAULT_FILLER_LIST, effective_filler_list(TIER_2),
            effective_filler_list(keep=["um"]))


def _encode(turns: Sequence[SpeakerTurn]) -> list[dict[str, Any]]:
    return [
        {"identity": list(turn.identity_key), "speaker": turn.speaker,
         "pieces": [[p.text, p.start_seconds, p.end_seconds, p.timed, p.segment_start,
                     p.rule.value, p.mapping.value,
                     [[str(a.segment_id), a.token_start, a.token_end, a.lex_start, a.lex_end]
                      for a in p.anchors]] for p in turn.pieces]}
        for turn in turns
    ]


def _decode(data: list[dict[str, Any]]) -> list[SpeakerTurn]:
    return [SpeakerTurn(tuple(t["identity"]), t["speaker"], tuple(
        TurnPiece(*p[:5], PieceRule(p[5]), TextMapping(p[6]), tuple(
            WordAnchor(uuid.UUID(a[0]), *a[1:]) for a in p[7])) for p in t["pieces"]
    )) for t in data]


def _render(
    turns: Sequence[SpeakerTurn], fillers: FillerList | None, repeats: bool,
) -> dict[str, Any]:
    result = apply_turn_filters(turns, fillers=fillers, drop_repeats=repeats)
    layout = layout_turns(result.turns)
    return {
        "turns": [[t.speaker, join_pieces(t.pieces)] for t in result.turns],
        "counts": [result.fillers_removed, result.repeats_removed],
        "seams": ([sorted(seams) for _, seams in drop_fillers_with_seams(turns, fillers=fillers)]
                  if fillers is not None else None),
        "reading": [[p.speaker, p.continuation, p.start_seconds,
                     [[r.marker_seconds, r.text] for r in p.runs]] for p in layout],
        "md": to_markdown_turns(layout, header="T"),
        "txt": to_txt_turns(layout),
    }


def _generated() -> list[tuple[str, list[SpeakerTurn]]]:
    rng = random.Random(75702)
    tokens = [*TIER_1, *TIER_2, '"Um,"', "(uh)", "'erm'", "erm...", "um's", "uh-huh",
              "the the", "we were we were", "café", "123", "3.14", "ß", "coil", "froze."]
    cases: list[tuple[str, list[SpeakerTurn]]] = []
    for case in range(500):
        turns: list[SpeakerTurn] = []
        identities = rng.randrange(1, 5)
        for turn_index in range(rng.randrange(1, 5)):
            name = f"Speaker {turn_index % identities}"
            text = " ".join(rng.choice([str.lower, str.upper, str.capitalize])(rng.choice(tokens))
                            + rng.choice(["", "", ",", ".", "?"])
                            for _ in range(rng.randrange(1, 4)))
            if case % 5 == 0:
                name = "Alex" if turn_index != 1 else "Sam"
                text = "um, uh." if turn_index == 1 else rng.choice(
                    ["the", "the coil", "Yes, um.", "you know, froze."])
            pieces: list[TurnPiece] = []
            units = list(re.finditer(r"\s*\S+", text))
            position = 0
            while position < len(units):
                size = 1 if rng.random() < 0.7 else rng.randrange(1, len(units) - position + 1)
                chunk = text[units[position].start():units[position + size - 1].end()]
                coarse = size > 1
                split = coarse and rng.choice([True, False])
                start = case % 3 * 58 + turn_index * 8 + position
                anchors = tuple(WordAnchor(
                    uuid.UUID(int=case * 10 + turn_index + 1), position + i, position + i + 1,
                    m.start(), m.end()) for i, m in enumerate(re.finditer(r"\S+", chunk)))
                pieces.append(TurnPiece(
                    chunk, start, start + 0.8, not coarse, position == 0 or rng.random() < 0.2,
                    PieceRule.SPLIT_CHILD if split else (
                        PieceRule.NO_WORDS if coarse else PieceRule.WORD_LEVEL),
                    TextMapping.COARSE if coarse else TextMapping.VERBATIM,
                    anchors if not coarse or split else (),
                ))
                position += size
            turns.append(SpeakerTurn(("speaker", name), name, tuple(pieces)))
        cases.append((f"generated:{case}", turns))
    return cases


def _configuration(fillers: FillerList | None, repeats: bool) -> dict[str, Any]:
    return {"fillers": asdict(fillers) if fillers is not None else None,
            "drop_repeats": repeats}


def _captured_render(turns: Sequence[SpeakerTurn], config: dict[str, Any]) -> dict[str, Any]:
    encoded = config["fillers"]
    fillers = FillerList(**{key: tuple(value) if isinstance(value, list) else value
                           for key, value in encoded.items()}) if encoded is not None else None
    return _render(turns, fillers, config["drop_repeats"])


def _existing_inputs() -> list[tuple[str, list[SpeakerTurn], dict[str, Any]]]:
    """Capture actual filter inputs and configurations from existing tests."""
    cases: list[tuple[str, list[SpeakerTurn], dict[str, Any]]] = []
    seen: set[str] = set()
    label = ""

    def capture(fn: Callable[..., Any], *, pieces: bool = False) -> Callable[..., Any]:
        def wrapped(value: Any, *args: Any, **kwargs: Any) -> Any:
            turns = [SpeakerTurn(("speaker", "Alex"), "Alex", value)] if pieces else value
            config = _configuration(kwargs.get("fillers"),
                                    kwargs.get("drop_repeats", fn.__name__ == "drop_repeats"))
            encoded = json.dumps([_encode(turns), config], sort_keys=True)
            if encoded not in seen:
                seen.add(encoded)
                cases.append((label, list(turns), config))
            return fn(value, *args, **kwargs)
        return wrapped

    for name in ("test_fillers", "test_repeats", "test_turn_filters"):
        module = importlib.import_module(f"tests.unit.{name}")
        targets = [key for key in ("drop_fillers", "drop_fillers_with_seams", "apply_turn_filters",
                                  "drop_repeats", "_clean_indexed") if hasattr(module, key)]
        with _capture_filters(module, targets, capture):
            for key, test in vars(module).items():
                if not key.startswith("test_") or not callable(test):
                    continue
                parameter_sets: list[list[dict[str, Any]]] = []
                for mark in getattr(test, "pytestmark", []):
                    if mark.name != "parametrize":
                        continue
                    names = [s.strip() for s in mark.args[0].split(",")]
                    parameter_sets.append([
                        dict(zip(names, (v,) if len(names) == 1 else v, strict=True))
                        for v in mark.args[1]
                    ])
                for index, combination in enumerate(itertools.product(*parameter_sets)):
                    label = f"{name}:{key}[{index}]"
                    test(**{k: v for row in combination for k, v in row.items()})
    # Existing tests iterate frozensets; assign ordinals after sorting their inputs.
    ordered = sorted(cases, key=lambda case: (
        case[0], json.dumps([_encode(case[1]), case[2]], sort_keys=True),
    ))
    ordinals: dict[str, int] = {}
    unique: list[tuple[str, list[SpeakerTurn], dict[str, Any]]] = []
    for name, turns, config in ordered:
        ordinal = ordinals.get(name, 0)
        unique.append((f"{name}:{ordinal}", turns, config))
        ordinals[name] = ordinal + 1
    return unique


def _capture_filters(
    module: ModuleType, targets: list[str], capture: Callable[..., Callable[..., Any]],
) -> ExitStack:
    stack = ExitStack()
    for key in targets:
        stack.enter_context(patch.object(module, key, capture(
            getattr(module, key), pieces=key == "_clean_indexed")))
    return stack


def _regenerate() -> None:
    cases = _existing_inputs() + [(name, turns, None) for name, turns in _generated()]
    cases += [(f"soup:{i}", [SpeakerTurn(("speaker", "Alex"), "Alex", pieces)], None)
              for i, pieces in enumerate(seeded_filler_soups(200))]
    # Pool identical option outputs per case, retaining readable output strings.
    lines = ['{', '  "version": 2,',
             '  "grid": ["none/false", "none/true", "default/false", "default/true",'
             ' "tier2/false", "tier2/true", "keep_um/false", "keep_um/true"],',
             '  "cases": [']
    for index, (name, turns, config) in enumerate(cases):
        outputs: list[dict[str, Any]] = []
        grid: list[int] = []
        for option, repeats in itertools.product(range(4), (False, True)):
            output = _render(turns, _OPTIONS[option], repeats)
            if output not in outputs:
                outputs.append(output)
            grid.append(outputs.index(output))
        def dump(value: Any) -> str:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        lines.extend(['    {"name": ' + dump(name) + ', "input": ' + dump(_encode(turns)) + ',',
                      '     "grid": ' + dump(grid) + ', "outputs": ['])
        lines.extend('       ' + dump(output) + (',' if i < len(outputs) - 1 else '')
                     for i, output in enumerate(outputs))
        if config is not None:
            captured = {"configuration": config, "output": _captured_render(turns, config)}
            lines.append('     ], "captured": ' + dump(captured) + '}'
                         + (',' if index < len(cases) - 1 else ''))
        else:
            lines.append('     ]}' + (',' if index < len(cases) - 1 else ''))
    lines.extend(['  ]', '}'])
    _FIXTURE.write_text("\n".join(lines) + "\n")
    print(f"{len(cases)} cases; {_FIXTURE.stat().st_size} bytes")


_DATA = json.loads(_FIXTURE.read_text()) if _FIXTURE.exists() else {"cases": []}


@pytest.mark.parametrize("case", _DATA["cases"], ids=lambda case: case["name"])
def test_filter_characterization(case: dict[str, Any]) -> None:
    turns = _decode(case["input"])
    for index, (option, repeats) in enumerate(itertools.product(range(4), (False, True))):
        # Normalize dataclass tuples to their JSON representation before comparison.
        actual = json.loads(json.dumps(_render(turns, _OPTIONS[option], repeats)))
        assert actual == case["outputs"][case["grid"][index]], _DATA["grid"][index]

    if "captured" in case:
        captured = case["captured"]
        actual = json.loads(json.dumps(_captured_render(turns, captured["configuration"])))
        assert actual == captured["output"], "captured configuration"


def test_characterization_fixture_present() -> None:
    assert len(_DATA["cases"]) >= 500
    generated = [c for c in _DATA["cases"] if c["name"].startswith("generated:")]
    assert len(generated) == 500
    assert max(len({tuple(t["identity"]) for t in c["input"]}) for c in generated) == 4
    names = [c["name"] for c in _DATA["cases"]]
    assert len(names) == len(set(names))
    assert sum(name.startswith("soup:") for name in names) == 200
    for entry in ("Weiß", "İ", "you know what"):
        harvested = [c for c in _DATA["cases"] if "captured" in c
                     and c["captured"]["configuration"]["fillers"] is not None
                     and entry in (*c["captured"]["configuration"]["fillers"]["words"],
                                   *c["captured"]["configuration"]["fillers"]["phrases"])]
        assert harvested, entry
        assert any(c["captured"]["output"]["counts"][0] > 0 for c in harvested), entry
    assert _FIXTURE.stat().st_size < 2_500_000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true", required=True)
    parser.parse_args()
    _regenerate()
