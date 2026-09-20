// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiFetch } from "../lib/api-client";
import type { MergeSuggestion } from "../lib/merge-candidates";
import { MergeSuggestionToast } from "./MergeSuggestionToast";
import type { LabelsResult } from "./SpeakerRail";

vi.mock("../lib/api-client", async (original) => ({
  ...(await original<typeof import("../lib/api-client")>()),
  apiFetch: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(apiFetch).mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const suggestion: MergeSuggestion = {
  candidateLabels: ["SPEAKER_01"],
  targetSpeakerId: "alice",
  targetSpeakerName: "Alice Chen",
  assignedLabel: "SPEAKER_00",
};

const preview = {
  labels: ["SPEAKER_00", "SPEAKER_01"],
  speakerId: "alice",
  speakerName: "Alice Chen",
  turnsMoved: 3,
  expected: { SPEAKER_00: "alice", SPEAKER_01: null },
};

function jsonResponse(data: unknown): Response {
  return { json: () => Promise.resolve(data) } as Response;
}

function setup(
  overrides: Partial<ComponentProps<typeof MergeSuggestionToast>> = {},
) {
  const props: ComponentProps<typeof MergeSuggestionToast> = {
    suggestion,
    runId: "run-1",
    reviewToken: "review-token",
    onClaimLost: vi.fn(),
    onMerged: vi.fn(),
    onDismiss: vi.fn(),
    ...overrides,
  };
  const result = render(<MergeSuggestionToast {...props} />);
  return { ...result, props };
}

describe("MergeSuggestionToast", () => {
  it("renders candidate label names and the target speaker name", () => {
    setup({
      suggestion: {
        ...suggestion,
        candidateLabels: ["SPEAKER_01", "SPEAKER_02"],
      },
    });
    expect(
      screen.getByText(
        "SPEAKER_01, SPEAKER_02 also sound like Alice Chen. Merge?",
      ),
    ).toBeTruthy();
  });

  it("previews then merges and passes the response to onMerged", async () => {
    const data: LabelsResult = {
      labels: [],
      segments: [],
      progress: { verified: 3, total: 3 },
    };
    vi.mocked(apiFetch)
      .mockResolvedValueOnce(jsonResponse(preview))
      .mockResolvedValueOnce(jsonResponse(data));
    const { props } = setup();

    fireEvent.click(screen.getByRole("button", { name: "Merge" }));

    await waitFor(() => {
      expect(props.onMerged).toHaveBeenCalledExactlyOnceWith(data);
    });
    expect(apiFetch).toHaveBeenCalledTimes(2);
    expect(apiFetch).toHaveBeenNthCalledWith(
      1,
      "/review/run-1/merge/preview",
      expect.objectContaining({ method: "POST" }),
    );
    expect(apiFetch).toHaveBeenNthCalledWith(
      2,
      "/review/run-1/merge",
      expect.objectContaining({ method: "POST" }),
    );
    const previewBody = new URLSearchParams(
      vi.mocked(apiFetch).mock.calls[0][1]?.body as string,
    );
    expect(previewBody.get("token")).toBe(props.reviewToken);
    expect(previewBody.get("target")).toBe(suggestion.targetSpeakerId);
    expect(previewBody.getAll("labels")).toEqual(preview.labels);
    const mergeBody = new URLSearchParams(
      vi.mocked(apiFetch).mock.calls[1][1]?.body as string,
    );
    expect(mergeBody.get("token")).toBe(props.reviewToken);
    expect(mergeBody.get("expected")).toBe(JSON.stringify(preview.expected));
    expect(mergeBody.get("speaker_id")).toBe(preview.speakerId);
    expect(mergeBody.getAll("labels")).toEqual(preview.labels);
    expect(mergeBody.get("nonce")).toBeTruthy();
    expect(props.onClaimLost).not.toHaveBeenCalled();
  });

  it.each(["preview", "merge"])(
    "shows the API error when %s fails without a conflict",
    async (stage) => {
      if (stage === "merge") {
        vi.mocked(apiFetch).mockResolvedValueOnce(jsonResponse(preview));
      }
      vi.mocked(apiFetch).mockRejectedValueOnce(
        new ApiError(500, "Unable to merge speakers."),
      );
      const { props } = setup();

      fireEvent.click(screen.getByRole("button", { name: "Merge" }));

      await waitFor(() => {
        expect(screen.getByText("Unable to merge speakers.")).toBeTruthy();
      });
      expect(props.onMerged).not.toHaveBeenCalled();
      expect(props.onClaimLost).not.toHaveBeenCalled();
      expect(props.onDismiss).not.toHaveBeenCalled();
    },
  );

  it.each(["preview", "merge"])(
    "dismisses without claim loss when %s returns a non-claim 409",
    async (stage) => {
      if (stage === "merge") {
        vi.mocked(apiFetch).mockResolvedValueOnce(jsonResponse(preview));
      }
      vi.mocked(apiFetch).mockRejectedValueOnce(
        new ApiError(409, "Conflict."),
      );
      const { props } = setup();

      fireEvent.click(screen.getByRole("button", { name: "Merge" }));

      await waitFor(() => {
        expect(props.onDismiss).toHaveBeenCalledOnce();
      });
      expect(props.onClaimLost).not.toHaveBeenCalled();
      expect(props.onMerged).not.toHaveBeenCalled();
    },
  );

  it.each(["preview", "merge"])(
    "calls onClaimLost when %s returns a claim-conflict 409",
    async (stage) => {
      if (stage === "merge") {
        vi.mocked(apiFetch).mockResolvedValueOnce(jsonResponse(preview));
      }
      vi.mocked(apiFetch).mockRejectedValueOnce(
        new ApiError(409, "Claim taken.", "claim"),
      );
      const { props } = setup();

      fireEvent.click(screen.getByRole("button", { name: "Merge" }));

      await waitFor(() => {
        expect(props.onClaimLost).toHaveBeenCalledOnce();
        expect(props.onDismiss).toHaveBeenCalledOnce();
      });
      expect(props.onMerged).not.toHaveBeenCalled();
    },
  );

  it("keeps the original deadline and uses the latest dismiss callback", () => {
    vi.useFakeTimers();
    const { props, rerender } = setup();
    act(() => vi.advanceTimersByTime(10_000));
    const onDismiss = vi.fn();
    rerender(<MergeSuggestionToast {...props} onDismiss={onDismiss} />);

    act(() => vi.advanceTimersByTime(5_000));

    expect(props.onDismiss).not.toHaveBeenCalled();
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  it("cancels auto-dismiss and prevents duplicate requests while merging", () => {
    vi.useFakeTimers();
    vi.mocked(apiFetch).mockReturnValue(new Promise<Response>(() => {}));
    const { props } = setup();
    const mergeButton = screen.getByRole("button", { name: "Merge" });

    act(() => {
      mergeButton.click();
      mergeButton.click();
    });
    act(() => vi.advanceTimersByTime(15_000));

    expect(apiFetch).toHaveBeenCalledTimes(1);
    expect(props.onDismiss).not.toHaveBeenCalled();
  });

  it("calls onDismiss when the dismiss button is clicked", () => {
    const { props } = setup();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(props.onDismiss).toHaveBeenCalledOnce();
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it.each([
    { candidateLabels: ["SPEAKER_01"], buttonName: "Merge" },
    { candidateLabels: ["SPEAKER_01", "SPEAKER_02"], buttonName: "Merge all" },
  ])(
    "shows $buttonName for $candidateLabels",
    ({ candidateLabels, buttonName }) => {
      setup({ suggestion: { ...suggestion, candidateLabels } });
      expect(screen.getByRole("button", { name: buttonName })).toBeTruthy();
    },
  );
});
