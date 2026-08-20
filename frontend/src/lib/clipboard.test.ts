import { afterEach, describe, expect, it, vi } from "vitest";

import { writeClipboard } from "./clipboard";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("writeClipboard", () => {
  it("writes the text and reports success when the clipboard API is present", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    await expect(writeClipboard("hello")).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
  });

  it("reports failure when navigator.clipboard is unavailable (plain-http LAN)", async () => {
    vi.stubGlobal("navigator", {});
    await expect(writeClipboard("hello")).resolves.toBe(false);
  });

  it("reports failure when writeText rejects, never throwing", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("denied"));
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    await expect(writeClipboard("hello")).resolves.toBe(false);
  });
});
