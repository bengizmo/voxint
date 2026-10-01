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
import type { LabelsResult } from "./SpeakerRail";
import { UndoToast } from "./UndoToast";

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
});

type UndoPayload = ComponentProps<typeof UndoToast>["undo"];

const labels: LabelsResult = {
  labels: [],
  segments: [],
  progress: { verified: 1, total: 3 },
};

function inMinutes(minutes: number): string {
  return new Date(Date.now() + minutes * 60_000).toISOString();
}

function jsonResponse(data: unknown): Response {
  return { json: () => Promise.resolve(data) } as Response;
}

function setup(overrides: Partial<ComponentProps<typeof UndoToast>> = {}) {
  const props: ComponentProps<typeof UndoToast> = {
    undo: { kind: "decide", decisionId: "dec-1", expiresAt: inMinutes(5) },
    runId: "run-1",
    reviewToken: "review-token",
    claimCsrf: "claim-csrf",
    onClaimLost: vi.fn(),
    onUndone: vi.fn(),
    onDismiss: vi.fn(),
    ...overrides,
  };
  const result = render(<UndoToast {...props} />);
  return { ...result, props };
}

function postedBody(call = 0): URLSearchParams {
  return new URLSearchParams(
    vi.mocked(apiFetch).mock.calls[call][1]?.body as string,
  );
}

describe("UndoToast", () => {
  it.each<[UndoPayload, string]>([
    [
      { kind: "decide", decisionId: "dec-1", expiresAt: inMinutes(5) },
      "Decision applied.",
    ],
    [
      { kind: "enroll", decisionId: "dec-1", expiresAt: inMinutes(5) },
      "Enrollment applied.",
    ],
    [
      { kind: "merge", mergeNonce: "merge-1", expiresAt: inMinutes(5) },
      "Labels merged.",
    ],
    [
      { kind: "relabel", decisionId: "dec-1", expiresAt: inMinutes(5) },
      "Segment speaker changed.",
    ],
  ])("labels a %o undo", (undo, text) => {
    setup({ undo });
    expect(screen.getByRole("status").textContent).toContain(text);
  });

  it.each<[UndoPayload, string, Record<string, string>]>([
    [
      { kind: "decide", decisionId: "dec-1", expiresAt: inMinutes(5) },
      "/review/run-1/undo/decide",
      { decision_id: "dec-1", nonce: "undo:dec-1" },
    ],
    [
      { kind: "enroll", decisionId: "dec-2", expiresAt: inMinutes(5) },
      "/review/run-1/undo/enroll",
      { decision_id: "dec-2", nonce: "undo:dec-2" },
    ],
    [
      { kind: "merge", mergeNonce: "merge-1", expiresAt: inMinutes(5) },
      "/review/run-1/undo/merge",
      { merge_nonce: "merge-1", nonce: "undo:merge-1" },
    ],
    [
      { kind: "relabel", decisionId: "dec-3", expiresAt: inMinutes(5) },
      "/review/run-1/undo/relabel",
      { decision_id: "dec-3", nonce: "undo:dec-3" },
    ],
  ])(
    "posts a %o undo to its endpoint and hands back the labels",
    async (undo, url, fields) => {
      vi.mocked(apiFetch).mockResolvedValueOnce(jsonResponse(labels));
      const { props } = setup({ undo });

      fireEvent.click(screen.getByRole("button", { name: "Undo" }));

      await waitFor(() => {
        expect(props.onUndone).toHaveBeenCalledExactlyOnceWith(labels);
      });
      expect(apiFetch).toHaveBeenCalledExactlyOnceWith(
        url,
        expect.objectContaining({
          method: "POST",
          headers: {
            "content-type": "application/x-www-form-urlencoded",
            accept: "application/json",
          },
        }),
      );
      const body = postedBody();
      expect(body.get("token")).toBe("review-token");
      expect(body.get("csrf_token")).toBe("claim-csrf");
      for (const [key, value] of Object.entries(fields)) {
        expect(body.get(key)).toBe(value);
      }
      expect(props.onClaimLost).not.toHaveBeenCalled();
    },
  );

  it("does nothing without a claim CSRF token", () => {
    setup({ claimCsrf: null });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("ignores a second click while the first undo is in flight", async () => {
    let resolve: (value: Response) => void = () => {};
    vi.mocked(apiFetch).mockReturnValueOnce(
      new Promise<Response>((r) => {
        resolve = r;
      }),
    );
    const { props } = setup();
    const button = screen.getByRole("button", { name: "Undo" });

    fireEvent.click(button);
    fireEvent.click(button);
    const busy = screen.getByRole<HTMLButtonElement>("button", {
      name: "Undoing…",
    });
    expect(busy.disabled).toBe(true);
    resolve(jsonResponse(labels));

    await waitFor(() => {
      expect(props.onUndone).toHaveBeenCalledOnce();
    });
    expect(apiFetch).toHaveBeenCalledOnce();
    const idle = screen.getByRole<HTMLButtonElement>("button", { name: "Undo" });
    expect(idle.disabled).toBe(false);
  });

  it("reports a drift 409 inside the window as too late to undo", async () => {
    // Pin the clock so a slow run cannot cross the deadline mid-test.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.mocked(apiFetch).mockRejectedValueOnce(new ApiError(409, "drift"));
    const { props } = setup();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain(
        "Too late to undo. The speaker was changed again since.",
      );
    });
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    expect(props.onClaimLost).not.toHaveBeenCalled();
    expect(props.onDismiss).not.toHaveBeenCalled();
    expect(props.onUndone).not.toHaveBeenCalled();
  });

  it("reports a 409 after the deadline as an expired window", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.mocked(apiFetch).mockImplementationOnce(() => {
      vi.setSystemTime(Date.now() + 6 * 60_000);
      return Promise.reject(new ApiError(409, "expired"));
    });
    setup();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain(
        "Undo window expired.",
      );
    });
  });

  it("stops the loop on a claim-conflict 409", async () => {
    vi.mocked(apiFetch).mockRejectedValueOnce(
      new ApiError(409, "Claim taken.", "claim"),
    );
    const { props } = setup();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(props.onClaimLost).toHaveBeenCalledOnce();
      expect(props.onDismiss).toHaveBeenCalledOnce();
    });
    expect(props.onUndone).not.toHaveBeenCalled();
  });

  it("shows the server detail for any other API error", async () => {
    vi.mocked(apiFetch).mockRejectedValueOnce(
      new ApiError(400, "use the merge undo endpoint"),
    );
    setup();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain(
        "use the merge undo endpoint",
      );
    });
  });

  it("falls back to a generic message on a network failure", async () => {
    vi.mocked(apiFetch).mockRejectedValueOnce(new TypeError("offline"));
    setup();

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain("Undo failed.");
    });
  });

  it("dismisses itself when the window closes", () => {
    vi.useFakeTimers();
    const { props } = setup({
      undo: {
        kind: "decide",
        decisionId: "dec-1",
        expiresAt: new Date(Date.now() + 30_000).toISOString(),
      },
    });

    act(() => vi.advanceTimersByTime(29_000));
    expect(props.onDismiss).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1_000));

    expect(props.onDismiss).toHaveBeenCalledOnce();
  });

  it("dismisses at once when the window already closed", () => {
    const { props } = setup({
      undo: {
        kind: "decide",
        decisionId: "dec-1",
        expiresAt: new Date(Date.now() - 1_000).toISOString(),
      },
    });

    expect(props.onDismiss).toHaveBeenCalledOnce();
  });

  it("dismisses from the close button", () => {
    const { props } = setup();

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));

    expect(props.onDismiss).toHaveBeenCalledOnce();
    expect(apiFetch).not.toHaveBeenCalled();
  });
});
