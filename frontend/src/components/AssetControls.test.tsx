// @vitest-environment jsdom
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AssetControls, type AssetControlsProps } from "./AssetControls";
import { OutlinePanel } from "./OutlinePanel";
import type { OutlineProps } from "../lib/outline";

const props: AssetControlsProps = {
  runId: "run-1",
  gatesOpen: true,
  sourceProblem: null,
  anyActive: false,
  csrfGenerate: "generate-token",
  csrfCancel: "cancel-token",
  kinds: ["summary", "topics", "entity_mentions"].map((kind) => ({
    kind,
    title: kind,
    hasAsset: false,
    stale: false,
    jobActive: false,
    jobId: null,
    jobStatus: null,
  })),
};
const outline: OutlineProps = {
  available: false,
  gated: false,
  assetStale: false,
  mentions: [],
  context: { summary: null, topics: [] },
  diagnostics: {
    droppedUnlocatable: 0,
    droppedOutOfRun: 0,
    droppedUnresolved: 0,
  },
};
function mockFetch(...results: unknown[]) {
  const fetch = vi.fn();
  for (const result of results)
    fetch.mockResolvedValueOnce(new Response(JSON.stringify(result)));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("asset controls", () => {
  it("gates generation and shows source problems", () => {
    const { rerender } = render(<AssetControls {...props} gatesOpen={false} />);
    expect(
      screen.getByText("Run assets are off. Enable in Settings → Features."),
    ).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
    rerender(<AssetControls {...props} sourceProblem="Transcript missing" />);
    expect(screen.getByText("Transcript missing")).toBeTruthy();
    expect(screen.queryByRole("button")).toBeNull();
  });
  it("posts all assets once while pending, then asks for reload", async () => {
    let resolve!: (response: Response) => void;
    const fetch = vi.fn(
      () =>
        new Promise<Response>((done) => {
          resolve = done;
        }),
    );
    vi.stubGlobal("fetch", fetch);
    render(<AssetControls {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Generate all" }));
    expect(screen.getByRole("status").textContent).toBe("Generating...");
    expect(screen.queryByRole("button")).toBeNull();
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/runs/run-1/assets/generate");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({
      Accept: "application/json",
      "Content-Type": "application/x-www-form-urlencoded",
    });
    expect(String(init.body)).toBe("csrf_token=generate-token");
    resolve(
      new Response(JSON.stringify({ started: true, error: null, created: 3 })),
    );
    expect(
      await screen.findByText("Generated. Reload to see updated content."),
    ).toBeTruthy();
    expect(fetch).toHaveBeenCalledTimes(1);
  });
  it("signals generation and restores controls when polling reports completion", async () => {
    mockFetch({ started: true, error: null, created: 1 });
    const onActive = vi.fn();
    const { rerender } = render(<AssetControls {...props} onActive={onActive} />);
    fireEvent.click(screen.getByRole("button", { name: "Generate all" }));
    await waitFor(() => expect(onActive).toHaveBeenCalledTimes(1));
    const polledState = {
      ...props,
      anyActive: true,
      kinds: [{
        ...props.kinds[0],
        jobActive: true,
        jobId: "job-1",
        jobStatus: "RUNNING",
      }],
    };
    rerender(<AssetControls {...props} polledState={polledState} onActive={onActive} />);
    expect(screen.getByRole("status").textContent).toBe("Generating...");
    expect(screen.getByText(/summary: RUNNING/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel summary" })).toBeTruthy();
    rerender(<AssetControls {...props} anyActive polledState={{
      ...polledState,
      anyActive: false,
      kinds: [{ ...props.kinds[0], hasAsset: true }],
    }} onActive={onActive} />);
    expect(screen.getByRole("status").textContent).toContain("Generated. Reload to see updated content.");
    expect(screen.getByRole("button", { name: "Regenerate all" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Cancel summary" })).toBeNull();
  });
  it("retries the same stale kind after a JSON-level error", async () => {
    const fetch = mockFetch(
      { started: false, error: "Worker unavailable", created: 0 },
      { started: true, error: null, created: 1 },
    );
    render(
      <AssetControls
        {...props}
        kinds={props.kinds.map((entry) => ({
          ...entry,
          hasAsset: true,
          stale: true,
        }))}
      />,
    );
    expect(screen.getByRole("button", { name: "Regenerate all" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Regenerate topics" }));
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "Worker unavailable Retry",
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await screen.findByText("Generated. Reload to see updated content.");
    for (const [, init] of fetch.mock.calls)
      expect(String(init.body)).toBe("csrf_token=generate-token&kind=topics");
  });
  it("surfaces HTTP errors from apiFetch", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "CSRF expired" }), {
            status: 403,
          }),
        ),
    );
    render(<AssetControls {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Generate all" }));
    expect((await screen.findByRole("alert")).textContent).toContain(
      "CSRF expired",
    );
  });
  it("retries failed cancellation and keeps cancellation pending until reload", async () => {
    const fetch = mockFetch({ cancelled: false }, { cancelled: true });
    render(
      <AssetControls
        {...props}
        anyActive
        kinds={[
          {
            ...props.kinds[0],
            jobActive: true,
            jobId: "job-1",
            jobStatus: "RUNNING",
          },
        ]}
      />,
    );
    expect(
      (
        screen.getByRole("button", {
          name: "Generate all",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.getByText(/summary: RUNNING/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Cancel summary" }));
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Cancel summary" }));
    await screen.findByText("Cancellation requested. Reload to check status.");
    expect(fetch.mock.calls[1][0]).toBe("/runs/run-1/assets/job-1/cancel");
    expect(String(fetch.mock.calls[1][1].body)).toBe("csrf_token=cancel-token");
    expect(
      (
        screen.getByRole("button", {
          name: "Cancel summary",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
  it("integrates empty and available outlines, preserves gating, and uses the panel runId", async () => {
    const fetch = mockFetch({ started: true, error: null, created: 3 });
    const panelProps = {
      runId: "editor-run",
      segments: [],
      capability: { seekEnabled: false, mediaDuration: null, reasons: [] },
      onJump: vi.fn(),
      assetControls: props,
    };
    const { rerender, container } = render(
      <OutlinePanel {...panelProps} outline={outline} />,
    );
    expect(
      screen.getByText("No outline was generated for this transcript."),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Generate all" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    expect(fetch.mock.calls[0][0]).toBe("/runs/editor-run/assets/generate");
    rerender(
      <OutlinePanel
        {...panelProps}
        outline={{ ...outline, available: true, assetStale: true }}
      />,
    );
    expect(
      container
        .querySelector("summary")
        ?.nextElementSibling?.getAttribute("aria-label"),
    ).toBe("Asset controls");
    expect(screen.getByText(/This outline was built/)).toBeTruthy();
    rerender(
      <OutlinePanel {...panelProps} outline={{ ...outline, gated: true }} />,
    );
    expect(container.innerHTML).toBe("");
  });
});
