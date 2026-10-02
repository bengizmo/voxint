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

  it("holds the editor write guard while the undo is in flight", async () => {
    let resolve: (value: Response) => void = () => {};
    vi.mocked(apiFetch).mockReturnValueOnce(
      new Promise<Response>((r) => {
        resolve = r;
      }),
    );
    const busyRef = { current: false };
    const setBusy = vi.fn();
    const { props } = setup({ writeGuard: { busy: false, busyRef, setBusy } });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(busyRef.current).toBe(true);
    expect(setBusy).toHaveBeenLastCalledWith(true);
    resolve(jsonResponse(labels));
    await waitFor(() => {
      expect(props.onUndone).toHaveBeenCalledOnce();
    });
    expect(busyRef.current).toBe(false);
    expect(setBusy).toHaveBeenLastCalledWith(false);
  });

  it("does not start while another editor write holds the guard", () => {
    const busyRef = { current: true };
    setup({ writeGuard: { busy: false, busyRef, setBusy: vi.fn() } });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    expect(apiFetch).not.toHaveBeenCalled();
    expect(busyRef.current).toBe(true);
  });

  it("disables Undo while another editor write is in flight", () => {
    const busyRef = { current: true };
    const { rerender, props } = setup({
      writeGuard: { busy: true, busyRef, setBusy: vi.fn() },
    });

    const undoButton = () =>
      screen.getByRole("button", { name: "Undo" }) as HTMLButtonElement;
    expect(undoButton().disabled).toBe(true);

    busyRef.current = false;
    rerender(
      <UndoToast
        {...props}
        writeGuard={{ busy: false, busyRef, setBusy: vi.fn() }}
      />,
    );
    expect(undoButton().disabled).toBe(false);
  });

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
        "Too late to undo. This was changed again since.",
      );
    });
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
    expect(props.onClaimLost).not.toHaveBeenCalled();
    expect(props.onDismiss).not.toHaveBeenCalled();
    expect(props.onUndone).not.toHaveBeenCalled();
  });

  it("explains an archived-speaker refusal, refetches, and lets the operator retry", async () => {
    const relabel: UndoPayload = {
      kind: "relabel",
      decisionId: "dec-9",
      expiresAt: inMinutes(5),
    };
    vi.mocked(apiFetch)
      .mockRejectedValueOnce(
        new ApiError(409, "the speaker ... is archived", "archived-speaker"),
      )
      .mockResolvedValueOnce(jsonResponse(labels));
    const onConflict = vi.fn(() => Promise.resolve());
    const { props } = setup({ undo: relabel, onConflict });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain(
        "Can't undo: the earlier speaker is archived. Restore them on the Speakers page, then undo again.",
      );
    });
    expect(screen.getByRole("status").textContent).not.toContain("Too late");
    expect(onConflict).toHaveBeenCalledOnce();
    expect(props.onDismiss).not.toHaveBeenCalled();

    // After restoring the speaker elsewhere, the same undo goes through.
    const retry = await screen.findByRole<HTMLButtonElement>("button", { name: "Undo" });
    expect(retry.disabled).toBe(false);
    fireEvent.click(retry);

    await waitFor(() => {
      expect(props.onUndone).toHaveBeenCalledWith(labels);
    });
    expect(apiFetch).toHaveBeenCalledTimes(2);
    for (const call of vi.mocked(apiFetch).mock.calls) {
      expect(call[0]).toBe("/review/run-1/undo/relabel");
      expect(String(call[1]?.body)).toContain("nonce=undo%3Adec-9");
    }
  });

  it("hides the retry button again when a retry fails for another reason", async () => {
    vi.mocked(apiFetch)
      .mockRejectedValueOnce(new ApiError(409, "archived", "archived-speaker"))
      .mockRejectedValueOnce(new ApiError(409, "drift"));
    setup({
      undo: { kind: "relabel", decisionId: "dec-9", expiresAt: inMinutes(5) },
    });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain("Too late to undo.");
    });
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
  });

  it("refetches through onConflict on a drift 409, still holding the write guard", async () => {
    vi.mocked(apiFetch).mockRejectedValueOnce(new ApiError(409, "drift"));
    const busyRef = { current: false };
    const setBusy = vi.fn();
    let finish: () => void = () => {};
    const onConflict = vi.fn(
      () =>
        new Promise<void>((r) => {
          finish = r;
        }),
    );
    const { props } = setup({
      onConflict,
      writeGuard: { busy: false, busyRef, setBusy },
    });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));

    await waitFor(() => {
      expect(onConflict).toHaveBeenCalledOnce();
    });
    // No other edit may start until the refetch has landed.
    expect(busyRef.current).toBe(true);
    expect(screen.getByRole("status").textContent).toContain("Too late to undo.");
    await act(async () => {
      finish();
    });
    expect(busyRef.current).toBe(false);
    expect(setBusy).toHaveBeenLastCalledWith(false);
    expect(props.onUndone).not.toHaveBeenCalled();
    expect(props.onDismiss).not.toHaveBeenCalled();
  });

  it.each<[string, ApiError | Error]>([
    ["a claim-conflict 409", new ApiError(409, "Claim taken.", "claim")],
    ["a 400", new ApiError(400, "use the merge undo endpoint")],
    ["a network failure", new TypeError("Failed to fetch")],
  ])("does not refetch after %s", async (_name, error) => {
    vi.mocked(apiFetch).mockRejectedValueOnce(error);
    const onConflict = vi.fn(() => Promise.resolve());
    const { props } = setup({ onConflict });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(screen.getByRole("button", { name: "Undoing…" })).toBeTruthy();

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Undoing…" })).toBeNull();
    });
    expect(onConflict).not.toHaveBeenCalled();
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

  it("keeps an undo failure on screen past the deadline", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    vi.mocked(apiFetch).mockRejectedValueOnce(new ApiError(409, "drift"));
    const { props } = setup({
      undo: {
        kind: "decide",
        decisionId: "dec-1",
        expiresAt: new Date(Date.now() + 30_000).toISOString(),
      },
    });

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.getByRole("status").textContent).toContain("Too late to undo.");
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });

    expect(props.onDismiss).not.toHaveBeenCalled();
    expect(screen.getByRole("status").textContent).toContain("Too late to undo.");
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
