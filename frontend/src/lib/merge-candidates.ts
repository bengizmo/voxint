import { isHumanRuling, type LabelStateShape } from "./speaker-bands";

export interface MergeSuggestion {
  candidateLabels: string[];
  targetSpeakerId: string;
  targetSpeakerName: string;
  assignedLabel: string;
}

export function findMergeCandidates(
  labels: LabelStateShape[],
  assignedLabel: string,
  assignedSpeakerId: string,
): MergeSuggestion | null {
  const candidates: string[] = [];
  let speakerName: string | null = null;

  for (const ls of labels) {
    if (ls.label === assignedLabel) {
      speakerName = ls.speakerName;
      continue;
    }
    if (isHumanRuling(ls)) continue;
    if (ls.resolution === "grounded_cosine") continue;
    if (ls.cosineSpeakerId !== assignedSpeakerId) continue;
    candidates.push(ls.label);
  }

  if (candidates.length === 0) return null;

  return {
    candidateLabels: candidates,
    targetSpeakerId: assignedSpeakerId,
    targetSpeakerName: speakerName ?? "speaker",
    assignedLabel,
  };
}
