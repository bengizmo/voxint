// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  SpeakerAssignPopover,
  type SpeakerAssignPopoverProps,
} from "./SpeakerAssignPopover";

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal("CSS", {
    escape: (value: string) => value.replaceAll(":", "\\:"),
  });
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function setup(overrides: Partial<SpeakerAssignPopoverProps> = {}) {
  const props: SpeakerAssignPopoverProps = {
    anchorRect: new DOMRect(40, 50, 100, 20),
    currentSpeaker: "Voice 3",
    currentSpeakerId: null,
    currentLabel: "SPEAKER_00",
    resolved: false,
    labelSegmentCount: 12,
    speakers: [{ id: "alice", displayName: "Alice Chen" }],
    onAssign: vi.fn(),
    onCreate: vi.fn(async () => true),
    onReset: vi.fn(),
    onRename: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };
  const result = render(<SpeakerAssignPopover {...props} />);
  return { ...result, props };
}

describe("SpeakerAssignPopover", () => {
  it("focuses search and assigns unresolved voices across their label", () => {
    const { props } = setup();
    expect(document.activeElement).toBe(screen.getByRole("combobox"));
    expect(
      (screen.getByRole("radio", { name: /All segments/ }) as HTMLInputElement)
        .checked,
    ).toBe(true);
    expect(screen.queryByRole("button", { name: /Rename/ })).toBeNull();
    fireEvent.click(screen.getByRole("option", { name: "Alice Chen" }));
    expect(props.onAssign).toHaveBeenCalledWith("alice", "label");
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("defaults resolved speakers to segment scope and permits switching before reset", () => {
    const { props } = setup({ resolved: true, currentSpeakerId: "alice" });
    expect(
      (
        screen.getByRole("radio", {
          name: "Just this segment",
        }) as HTMLInputElement
      ).checked,
    ).toBe(true);
    fireEvent.click(screen.getByRole("radio", { name: /All segments/ }));
    fireEvent.click(screen.getByRole("button", { name: /Reset/ }));
    expect(props.onReset).toHaveBeenCalledWith("label");
  });

  it("renames through the parent callback", () => {
    const { props } = setup({ resolved: true, currentSpeakerId: "alice" });
    fireEvent.click(screen.getByRole("button", { name: /Rename/ }));
    const input = screen.getByRole("textbox", { name: /Rename/ });
    expect(document.activeElement).toBe(input);
    fireEvent.change(input, { target: { value: " Mara " } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(props.onRename).toHaveBeenCalledWith("Mara");
  });

  it("keeps failed creations open and closes after successful creation", async () => {
    const onCreate = vi
      .fn()
      .mockResolvedValueOnce(false)
      .mockResolvedValueOnce(true);
    const { props } = setup({ onCreate });
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "Mara" },
    });
    await act(async () =>
      fireEvent.click(screen.getByRole("option", { name: 'Create "Mara"' })),
    );
    expect(props.onClose).not.toHaveBeenCalled();
    await act(async () =>
      fireEvent.click(screen.getByRole("option", { name: 'Create "Mara"' })),
    );
    expect(onCreate).toHaveBeenLastCalledWith("Mara", "label");
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("handles Escape from the nested combobox and restores the opener on unmount", () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    const { props, unmount } = setup();
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledOnce();
    unmount();
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });

  it("closes only for outside clicks, not internal interactions", () => {
    const { props } = setup();
    fireEvent.click(screen.getByRole("radio", { name: "Just this segment" }));
    expect(props.onClose).not.toHaveBeenCalled();
    fireEvent.click(document.body);
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("flips above near the bottom and clamps horizontally", () => {
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(
      new DOMRect(0, 0, 320, 300),
    );
    setup({
      anchorRect: new DOMRect(
        window.innerWidth - 20,
        window.innerHeight - 40,
        10,
        20,
      ),
    });
    const panel = screen.getByRole("dialog");
    expect(panel.style.top).toBe(`${window.innerHeight - 344}px`);
    expect(panel.style.left).toBe(`${window.innerWidth - 328}px`);
  });
});
