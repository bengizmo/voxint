export interface LabelStateShape {
  label: string;
  paletteIndex: number | null;
  turnCount: number;
  totalSeconds: number;
  resolution: string;
  speakerId: string | null;
  speakerName: string | null;
  cosineConfidence: number | null;
  cosineSpeakerId: string | null;
  cosineSpeakerName: string | null;
  cosineGrounded: boolean;
  llmHintName: string | null;
  band: string | null;
  bandReason: string | null;
  candidatePromptAllowed: boolean;
  candidateSpeakerId: string | null;
  candidateSpeakerName: string | null;
  matchDecision: string | null;
  matchReason: string | null;
  matchSimilarity: number | null;
  matchMargin: number | null;
  matchVoteAgreement: number | null;
  matchEligibleSeconds: number;
}

export type Group = "needsYou" | "tooShort" | "matchedAutomatically" | "yourRulings";

export const TOO_SHORT_REASONS: ReadonlySet<string> = new Set([
  "too_few_turns",
  "too_little_speech",
  "no_eligible_turns",
]);

export function isHumanRuling(state: LabelStateShape): boolean {
  return ["human_assign", "human_exclude", "human_unknown"].includes(state.resolution);
}

export function isConfirmable(state: LabelStateShape): boolean {
  return (
    !isHumanRuling(state) &&
    state.candidateSpeakerId !== null &&
    state.candidatePromptAllowed
  );
}

export function isAmbiguous(state: LabelStateShape): boolean {
  return !isHumanRuling(state) && state.band === "review" && !state.candidatePromptAllowed;
}

export function groupOf(state: LabelStateShape): Group {
  if (isHumanRuling(state)) {
    return "yourRulings";
  }
  if (state.resolution === "grounded_cosine") {
    return "matchedAutomatically";
  }
  if (state.resolution === "auto_enroll") {
    return isConfirmable(state) ? "needsYou" : "matchedAutomatically";
  }
  if (!isConfirmable(state) && state.matchReason !== null && TOO_SHORT_REASONS.has(state.matchReason)) {
    return "tooShort";
  }
  return "needsYou";
}

export interface RailPartition {
  needsYou: LabelStateShape[];
  tooShort: LabelStateShape[];
  matchedAutomatically: LabelStateShape[];
  yourRulings: LabelStateShape[];
}

function needsYouRank(state: LabelStateShape): number {
  if (isConfirmable(state) && state.resolution === "unresolved") {
    return 0;
  }
  if (isConfirmable(state) && state.resolution === "auto_enroll") {
    return 1;
  }
  if (isAmbiguous(state)) {
    return 2;
  }
  return 3;
}

export function partition(states: LabelStateShape[]): RailPartition {
  const result: RailPartition = {
    needsYou: [],
    tooShort: [],
    matchedAutomatically: [],
    yourRulings: [],
  };

  for (const state of states) {
    result[groupOf(state)].push(state);
  }

  result.needsYou.sort(
    (left, right) => needsYouRank(left) - needsYouRank(right) || left.label.localeCompare(right.label),
  );
  result.tooShort.sort((left, right) => left.label.localeCompare(right.label));
  result.matchedAutomatically.sort((left, right) => left.label.localeCompare(right.label));
  result.yourRulings.sort((left, right) => left.label.localeCompare(right.label));
  return result;
}

export interface Headline {
  title: string;
  detail: string;
  linkSpeakerId: string | null;
}

function voiceName(name: string | null): string {
  return name ?? "this voice";
}

export function headline(state: LabelStateShape): Headline {
  if (isConfirmable(state) && state.resolution === "unresolved") {
    return {
      title: `Possibly ${voiceName(state.candidateSpeakerName)}`,
      detail: state.bandReason ?? "Not strong enough to confirm without your check.",
      linkSpeakerId: state.candidateSpeakerId,
    };
  }
  if (isConfirmable(state) && state.resolution === "auto_enroll") {
    return {
      title: `Possibly ${voiceName(state.candidateSpeakerName)}`,
      detail: `Saved as ${voiceName(state.speakerName)} for now. Confirm if this is ${voiceName(state.candidateSpeakerName)}.`,
      linkSpeakerId: state.candidateSpeakerId,
    };
  }
  if (isAmbiguous(state)) {
    return {
      title: "Similar voices found",
      detail: "Two known speakers sound close to this voice. Listen, then choose.",
      linkSpeakerId: null,
    };
  }
  if (state.resolution === "grounded_cosine") {
    return {
      title: `Shown as ${voiceName(state.speakerName)}`,
      detail: "Matched by voice.",
      linkSpeakerId: state.speakerId,
    };
  }
  if (state.resolution === "auto_enroll") {
    return {
      title: `Saved as ${voiceName(state.speakerName)}`,
      detail: "Saved automatically so Voxint can recognise this voice later. Name them on the Speakers page.",
      linkSpeakerId: state.speakerId,
    };
  }
  if (state.resolution === "human_assign") {
    return {
      title: voiceName(state.speakerName),
      detail: "Your ruling.",
      linkSpeakerId: state.speakerId,
    };
  }
  if (state.resolution === "human_exclude") {
    return { title: "Left out", detail: "Your ruling: not a person.", linkSpeakerId: null };
  }
  if (state.resolution === "human_unknown") {
    return { title: "Could not tell", detail: "Your ruling.", linkSpeakerId: null };
  }
  return {
    title: "Who is this?",
    detail: state.bandReason ?? "Not enough evidence to suggest a speaker.",
    linkSpeakerId: null,
  };
}

export type Coverage = { kind: "not_run" } | { kind: "no_roster" } | { kind: "ok" };

export function coverage(states: LabelStateShape[]): Coverage {
  if (states.length > 0 && states.every((state) => state.matchDecision === null)) {
    return { kind: "not_run" };
  }
  if (
    states.some((state) => state.matchReason === "no_roster") &&
    !states.some((state) => state.candidateSpeakerId !== null)
  ) {
    return { kind: "no_roster" };
  }
  return { kind: "ok" };
}

export function summary(rail: RailPartition, matchCoverage: Coverage): string {
  const needsYou = rail.needsYou.length + rail.tooShort.length;
  if (needsYou === 0) {
    return "Every voice has a ruling.";
  }
  if (matchCoverage.kind === "not_run") {
    return "Voice matching did not run on this recording.";
  }
  if (matchCoverage.kind === "no_roster") {
    return "No known speakers were on your roster when this recording was matched. Add people below so Voxint can recognise their voices on later recordings.";
  }
  const short = rail.tooShort.length
    ? `, ${rail.tooShort.length} with very little speech`
    : "";
  return `${needsYou} voice${needsYou === 1 ? "" : "s"} need${needsYou === 1 ? "s" : ""} you${short}. ${rail.matchedAutomatically.length} matched automatically.`;
}

export function whyText(state: LabelStateShape): string | null {
  if (isAmbiguous(state)) {
    return "Two known speakers sound close to this voice. Listen, then choose.";
  }
  const similarity = state.matchSimilarity ?? state.cosineConfidence;
  if (similarity === null) {
    return null;
  }
  const parts = state.bandReason === null ? [] : [state.bandReason];
  parts.push(`Voice similarity ${similarity.toFixed(2)}.`);
  if (state.matchMargin !== null) {
    parts.push(`Lead over the next closest voice ${state.matchMargin.toFixed(2)}.`);
  } else if (state.matchDecision !== null) {
    // A recorded match with no margin means a one-speaker roster; a run with
    // no match row at all says nothing about the roster.
    parts.push("Lead over the next closest voice none (only one known speaker).");
  }
  if (state.matchVoteAgreement !== null) {
    parts.push(`Agreement across this voice's speech ${state.matchVoteAgreement.toFixed(2)}.`);
  }
  return parts.join(" ").trim();
}
