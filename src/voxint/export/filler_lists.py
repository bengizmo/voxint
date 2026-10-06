"""English filler presets and validated operator entries."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

PRESET_VERSION = "en-1"
TIER_1 = ("um", "uh", "umm", "uhh", "uhm", "erm")
TIER_2 = (
    "hm",
    "mm",
    "mmm",
    "you know",
    "kind of",
    "sort of",
    "I mean",
    "I guess",
    "I suppose",
    "or something",
    "you know what I mean",
    "you see",
)
NEVER = ("like", "so", "right", "well")
Source = Literal["settings", "environment", "none"]


class FillerListError(ValueError):
    """An invalid filler list, safe to describe to the operator."""


def normalize_entries(lines: Iterable[str]) -> tuple[str, ...]:
    """Validate entries and retain the first spelling of each identity."""
    entries: dict[str, str] = {}
    for line in lines:
        entry = " ".join(line.split())
        if not entry:
            continue
        if len(entry) > 40 or len(entry.split()) > 5:
            raise FillerListError("Use at most 5 words and 40 characters per entry.")
        for word in entry.split():
            for index, char in enumerate(word):
                previous = word[index - 1] if index else ""
                following = word[index + 1] if index + 1 < len(word) else ""
                if char.isalpha():
                    continue
                if (
                    char in "'\u2019"
                    and previous.isalpha()
                    and (not following or following.isalpha())
                ):
                    continue
                if char == "-" and previous.isalpha() and following.isalpha():
                    continue
                raise FillerListError(
                    "Use letters, apostrophes inside or after words, "
                    "and single hyphens between letters."
                )
        entries.setdefault(entry.casefold(), entry)
        if len(entries) > 100:
            raise FillerListError("Use at most 100 different entries per list.")
    return tuple(entries.values())


@dataclass(frozen=True)
class FillerList:
    language: str
    preset_version: str
    words: tuple[str, ...]
    phrases: tuple[str, ...]
    added: tuple[str, ...]
    kept: tuple[str, ...]
    kept_without_effect: tuple[str, ...]
    add_source: Source
    keep_source: Source
    env_overridden: tuple[str, ...]

    @property
    def extra_entries(self) -> tuple[str, ...]:
        effective = {entry.casefold() for entry in (*self.words, *self.phrases)}
        preset = {entry.casefold() for entry in TIER_1}
        return tuple(
            entry for entry in self.added
            if entry.casefold() in effective and entry.casefold() not in preset
        )

    @property
    def effective_keeps(self) -> tuple[str, ...]:
        ineffective = {entry.casefold() for entry in self.kept_without_effect}
        return tuple(entry for entry in self.kept if entry.casefold() not in ineffective)

    @property
    def is_default(self) -> bool:
        return {entry.casefold() for entry in (*self.words, *self.phrases)} == {
            entry.casefold() for entry in TIER_1
        }


def effective_filler_list(
    add: Iterable[str] = (),
    keep: Iterable[str] = (),
    *,
    add_source: Source = "none",
    keep_source: Source = "none",
    env_overridden: tuple[str, ...] = (),
) -> FillerList:
    """Resolve exact-entry keeps against the preset and additions."""
    added, kept = normalize_entries(add), normalize_entries(keep)
    # Casefold decides identity; matching keeps the first spelling, because
    # IGNORECASE cannot match an expanded fold (weiß is not weiss, nor İ i̇).
    available = {entry.casefold(): entry for entry in (*TIER_1, *added)}
    removed = {entry.casefold() for entry in kept}
    effective = [spelling for key, spelling in available.items() if key not in removed]
    return FillerList(
        language="en",
        preset_version=PRESET_VERSION,
        words=tuple(
            sorted(
                (entry for entry in effective if " " not in entry),
                key=lambda entry: (entry.casefold(), entry),
            )
        ),
        phrases=tuple(
            sorted(
                (entry for entry in effective if " " in entry),
                key=lambda entry: (-len(entry.split()), -len(entry), entry.casefold(), entry),
            )
        ),
        added=added,
        kept=kept,
        kept_without_effect=tuple(entry for entry in kept if entry.casefold() not in available),
        add_source=add_source,
        keep_source=keep_source,
        env_overridden=env_overridden,
    )


DEFAULT_FILLER_LIST = effective_filler_list()
# Every preset entry kept: the filler pass applies omit marks and nothing else.
NO_FILLER_LIST = effective_filler_list(keep=TIER_1)
