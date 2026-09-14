import { describe, expect, it } from "vitest";

import {
  coverage,
  groupOf,
  headline,
  isAmbiguous,
  isConfirmable,
  isHumanRuling,
  partition,
  summary,
  TOO_SHORT_REASONS,
  whyText,
  type LabelStateShape,
  type RailPartition,
} from "./speaker-bands";

function make(overrides: Partial<LabelStateShape> = {}): LabelStateShape {
  return {
    label: "Speaker 1",
    paletteIndex: null,
    turnCount: 3,
    totalSeconds: 20,
    resolution: "unresolved",
    speakerId: null,
    speakerName: null,
    cosineConfidence: null,
    cosineSpeakerId: null,
    cosineSpeakerName: null,
    cosineGrounded: false,
    llmHintName: null,
    band: null,
    bandReason: null,
    candidatePromptAllowed: false,
    candidateSpeakerId: null,
    candidateSpeakerName: null,
    matchDecision: "ineligible",
    matchReason: null,
    matchSimilarity: null,
    matchMargin: null,
    matchVoteAgreement: null,
    matchEligibleSeconds: 0,
    ...overrides,
  };
}

function emptyPartition(overrides: Partial<RailPartition> = {}): RailPartition {
  return {
    needsYou: [],
    tooShort: [],
    matchedAutomatically: [],
    yourRulings: [],
    ...overrides,
  };
}

describe("classification", () => {
  it("recognises human rulings, confirmable states, and ambiguous states", () => {
    for (const resolution of ["human_assign", "human_exclude", "human_unknown"]) {
      expect(isHumanRuling(make({ resolution }))).toBe(true);
    }
    expect(isHumanRuling(make({ resolution: "unresolved" }))).toBe(false);
    expect(
      isConfirmable(
        make({
          band: "auto_attribute",
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
        }),
      ),
    ).toBe(true);
    expect(
      isConfirmable(
        make({
          resolution: "human_assign",
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
        }),
      ),
    ).toBe(false);
    expect(isAmbiguous(make({ band: "review" }))).toBe(true);
    expect(isAmbiguous(make({ band: "review", candidatePromptAllowed: true }))).toBe(false);
    expect(isAmbiguous(make({ band: "review", resolution: "human_unknown" }))).toBe(false);
  });

  it("contains the three too-short reasons", () => {
    expect([...TOO_SHORT_REASONS]).toEqual([
      "too_few_turns",
      "too_little_speech",
      "no_eligible_turns",
    ]);
  });

  it("groups every resolution and unknown resolutions", () => {
    expect(groupOf(make({ resolution: "human_assign" }))).toBe("yourRulings");
    expect(groupOf(make({ resolution: "human_exclude" }))).toBe("yourRulings");
    expect(groupOf(make({ resolution: "human_unknown" }))).toBe("yourRulings");
    expect(groupOf(make({ resolution: "grounded_cosine" }))).toBe("matchedAutomatically");
    expect(groupOf(make({ resolution: "auto_enroll" }))).toBe("matchedAutomatically");
    expect(
      groupOf(
        make({
          resolution: "auto_enroll",
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
        }),
      ),
    ).toBe("needsYou");
    expect(
      groupOf(
        make({
          band: "auto_attribute",
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
        }),
      ),
    ).toBe("needsYou");
    expect(groupOf(make({ matchReason: "too_few_turns" }))).toBe("tooShort");
    expect(
      groupOf(
        make({
          matchReason: "too_few_turns",
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
        }),
      ),
    ).toBe("needsYou");
    expect(groupOf(make({ resolution: "future_resolution" }))).toBe("needsYou");
  });
});

describe("partition", () => {
  it("sorts all needs-you ranks and alphabetises every group", () => {
    const result = partition([
      make({ label: "Zulu", resolution: "human_assign" }),
      make({ label: "Bravo", band: "review" }),
      make({ label: "Alpha", resolution: "grounded_cosine" }),
      make({ label: "Delta" }),
      make({ label: "Charlie", matchReason: "too_little_speech" }),
      make({
        label: "Beta",
        resolution: "auto_enroll",
        candidatePromptAllowed: true,
        candidateSpeakerId: "known-1",
      }),
      make({
        label: "Able",
        candidatePromptAllowed: true,
        candidateSpeakerId: "known-1",
      }),
      make({ label: "Echo", resolution: "auto_enroll" }),
      make({ label: "Alpha", resolution: "human_unknown" }),
      make({ label: "Able", matchReason: "no_eligible_turns" }),
    ]);

    expect(result.needsYou.map((state) => state.label)).toEqual([
      "Able",
      "Beta",
      "Bravo",
      "Delta",
    ]);
    expect(result.tooShort.map((state) => state.label)).toEqual(["Able", "Charlie"]);
    expect(result.matchedAutomatically.map((state) => state.label)).toEqual(["Alpha", "Echo"]);
    expect(result.yourRulings.map((state) => state.label)).toEqual(["Alpha", "Zulu"]);
  });
});

describe("headline", () => {
  it("describes unresolved confirmable states with custom and fallback detail", () => {
    const confirmable = {
      candidatePromptAllowed: true,
      candidateSpeakerId: "known-1",
      candidateSpeakerName: "Ada",
    };
    expect(headline(make({ ...confirmable, bandReason: "Needs review." }))).toEqual({
      title: "Possibly Ada",
      detail: "Needs review.",
    });
    expect(headline(make(confirmable))).toEqual({
      title: "Possibly Ada",
      detail: "Not strong enough to confirm without your check.",
    });
  });

  it("describes confirmable and automatic enrolment", () => {
    expect(
      headline(
        make({
          resolution: "auto_enroll",
          speakerName: "Voice 2",
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
          candidateSpeakerName: "Ada",
        }),
      ),
    ).toEqual({
      title: "Possibly Ada",
      detail: "Saved as Voice 2 for now. Confirm if this is Ada.",
    });
    expect(headline(make({ resolution: "auto_enroll", speakerName: "Voice 2" }))).toEqual({
      title: "Saved as Voice 2",
      detail:
        "Saved automatically so Voxint can recognise this voice later. Name them on the Speakers page.",
    });
  });

  it("describes ambiguous and grounded matches", () => {
    expect(headline(make({ band: "review" }))).toEqual({
      title: "Similar voices found",
      detail: "Two known speakers sound close to this voice. Listen, then choose.",
    });
    expect(
      headline(make({ resolution: "grounded_cosine", speakerName: "Grace" })),
    ).toEqual({ title: "Shown as Grace", detail: "Matched by voice." });
  });

  it("describes every human ruling", () => {
    expect(headline(make({ resolution: "human_assign", speakerName: "Lin" }))).toEqual({
      title: "Lin",
      detail: "Your ruling.",
    });
    expect(headline(make({ resolution: "human_exclude" }))).toEqual({
      title: "Left out",
      detail: "Your ruling: not a person.",
    });
    expect(headline(make({ resolution: "human_unknown" }))).toEqual({
      title: "Could not tell",
      detail: "Your ruling.",
    });
  });

  it("describes unresolved fallback states and substitutes missing names", () => {
    expect(headline(make({ bandReason: "No eligible speech." }))).toEqual({
      title: "Who is this?",
      detail: "No eligible speech.",
    });
    expect(headline(make())).toEqual({
      title: "Who is this?",
      detail: "Not enough evidence to suggest a speaker.",
    });
    expect(
      headline(
        make({
          candidatePromptAllowed: true,
          candidateSpeakerId: "known-1",
          candidateSpeakerName: null,
        }),
      ).title,
    ).toBe("Possibly this voice");
    expect(headline(make({ resolution: "grounded_cosine" })).title).toBe(
      "Shown as this voice",
    );
    expect(headline(make({ resolution: "human_assign" })).title).toBe("this voice");
  });
});

describe("coverage", () => {
  it("reports matching that did not run", () => {
    expect(coverage([make({ matchDecision: null }), make({ matchDecision: null })])).toEqual({
      kind: "not_run",
    });
  });

  it("reports a missing roster among other ineligible labels", () => {
    expect(
      coverage([
        make({ matchReason: "no_roster" }),
        make({ matchReason: "too_few_turns" }),
      ]),
    ).toEqual({ kind: "no_roster" });
  });

  it("reports ok for candidates, ordinary evidence, and empty input", () => {
    expect(
      coverage([
        make({ matchReason: "no_roster", candidateSpeakerId: "known-1" }),
      ]),
    ).toEqual({ kind: "ok" });
    expect(coverage([make()])).toEqual({ kind: "ok" });
    expect(coverage([])).toEqual({ kind: "ok" });
  });
});

describe("summary", () => {
  it("handles singular and plural needs", () => {
    expect(summary(emptyPartition({ needsYou: [make()] }), { kind: "ok" })).toBe(
      "1 voice needs you. 0 matched automatically.",
    );
    expect(
      summary(
        emptyPartition({
          needsYou: [make(), make()],
          matchedAutomatically: [make(), make(), make()],
        }),
        { kind: "ok" },
      ),
    ).toBe("2 voices need you. 3 matched automatically.");
  });

  it("includes voices with very little speech", () => {
    expect(
      summary(
        emptyPartition({
          needsYou: [make()],
          tooShort: [make()],
          matchedAutomatically: [make(), make(), make()],
        }),
        { kind: "ok" },
      ),
    ).toBe("2 voices need you, 1 with very little speech. 3 matched automatically.");
  });

  it("reports the finish line", () => {
    expect(summary(emptyPartition({ matchedAutomatically: [make()] }), { kind: "ok" })).toBe(
      "Every voice has a ruling.",
    );
  });

  it("uses coverage override messages", () => {
    expect(summary(emptyPartition(), { kind: "not_run" })).toBe(
      "Voice matching did not run on this recording.",
    );
    expect(summary(emptyPartition(), { kind: "no_roster" })).toBe(
      "No known speakers to match against yet. Add people below so Voxint can recognise their voices on later recordings.",
    );
  });
});

describe("whyText", () => {
  it("uses the fixed ambiguous explanation without evidence values", () => {
    expect(
      whyText(
        make({
          band: "review",
          candidateSpeakerName: "Ada",
          matchSimilarity: 0.91,
          matchMargin: 0.02,
        }),
      ),
    ).toBe("Two known speakers sound close to this voice. Listen, then choose.");
  });

  it("returns null without similarity evidence", () => {
    expect(whyText(make())).toBeNull();
  });

  it("describes one-speaker evidence without agreement", () => {
    expect(
      whyText(make({ bandReason: "Strong match.", cosineConfidence: 0.876, matchMargin: null })),
    ).toBe(
      "Strong match. Voice similarity 0.88. Lead over the next closest voice none (only one known speaker).",
    );
  });

  it("describes match evidence with margin and agreement", () => {
    expect(
      whyText(
        make({
          cosineConfidence: 0.2,
          matchSimilarity: 0.934,
          matchMargin: 0.127,
          matchVoteAgreement: 0.756,
        }),
      ),
    ).toBe(
      "Voice similarity 0.93. Lead over the next closest voice 0.13. Agreement across this voice's speech 0.76.",
    );
  });
});
