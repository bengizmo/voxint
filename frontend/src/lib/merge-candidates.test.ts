import { describe, expect, it } from "vitest";

import type { LabelStateShape } from "./speaker-bands";
import { findMergeCandidates } from "./merge-candidates";

function make(overrides: Partial<LabelStateShape> = {}): LabelStateShape {
  return {
    label: "SPEAKER_00",
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

const SPEAKER_ID = "aaa-111";

describe("findMergeCandidates", () => {
  it("returns null when no labels match the assigned speaker", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: "Alice" }),
      make({ label: "SPEAKER_01", cosineSpeakerId: "other-id" }),
      make({ label: "SPEAKER_02", cosineSpeakerId: null }),
    ];
    expect(findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID)).toBeNull();
  });

  it("returns candidates for unresolved labels with matching cosineSpeakerId", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: "Alice" }),
      make({ label: "SPEAKER_01", cosineSpeakerId: SPEAKER_ID }),
      make({ label: "SPEAKER_02", cosineSpeakerId: SPEAKER_ID }),
    ];
    const result = findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID);
    expect(result).toEqual({
      candidateLabels: ["SPEAKER_01", "SPEAKER_02"],
      targetSpeakerId: SPEAKER_ID,
      targetSpeakerName: "Alice",
      assignedLabel: "SPEAKER_00",
    });
  });

  it("excludes labels with human rulings", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: "Alice" }),
      make({ label: "SPEAKER_01", resolution: "human_assign", cosineSpeakerId: SPEAKER_ID }),
      make({ label: "SPEAKER_02", resolution: "human_exclude", cosineSpeakerId: SPEAKER_ID }),
      make({ label: "SPEAKER_03", resolution: "human_unknown", cosineSpeakerId: SPEAKER_ID }),
      make({ label: "SPEAKER_04", cosineSpeakerId: SPEAKER_ID }),
    ];
    const result = findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID);
    expect(result?.candidateLabels).toEqual(["SPEAKER_04"]);
  });

  it("excludes grounded_cosine labels", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: "Alice" }),
      make({ label: "SPEAKER_01", resolution: "grounded_cosine", cosineSpeakerId: SPEAKER_ID }),
      make({ label: "SPEAKER_02", cosineSpeakerId: SPEAKER_ID }),
    ];
    const result = findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID);
    expect(result?.candidateLabels).toEqual(["SPEAKER_02"]);
  });

  it("excludes the assigned label itself", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: "Alice", cosineSpeakerId: SPEAKER_ID }),
    ];
    expect(findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID)).toBeNull();
  });

  it("includes auto_enroll labels", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: "Alice" }),
      make({ label: "SPEAKER_01", resolution: "auto_enroll", cosineSpeakerId: SPEAKER_ID }),
    ];
    const result = findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID);
    expect(result?.candidateLabels).toEqual(["SPEAKER_01"]);
  });

  it("falls back to 'speaker' when speakerName is null", () => {
    const labels = [
      make({ label: "SPEAKER_00", resolution: "human_assign", speakerId: SPEAKER_ID, speakerName: null }),
      make({ label: "SPEAKER_01", cosineSpeakerId: SPEAKER_ID }),
    ];
    const result = findMergeCandidates(labels, "SPEAKER_00", SPEAKER_ID);
    expect(result?.targetSpeakerName).toBe("speaker");
  });
});
