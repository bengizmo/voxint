// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SpeakerCombobox } from "./SpeakerCombobox";

Element.prototype.scrollIntoView ??= () => {};

beforeEach(() => {
  vi.stubGlobal("CSS", { escape: (value: string) => value });
  vi.spyOn(Element.prototype, "scrollIntoView").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

const speakers = [{ id: "alice", displayName: "Alice Chen" }];

describe("SpeakerCombobox outside press", () => {
  it("closes the list on a press outside it by default", () => {
    render(
      <SpeakerCombobox
        speakers={speakers}
        mode="controlled"
        label="Choose speaker"
        onSelect={vi.fn()}
        autoFocus
      />,
    );
    expect(screen.getByRole("listbox")).toBeTruthy();

    fireEvent.mouseDown(document.body);

    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("keeps the list open when the host opts out", () => {
    render(
      <SpeakerCombobox
        speakers={speakers}
        mode="controlled"
        label="Choose speaker"
        onSelect={vi.fn()}
        autoFocus
        dismissOnOutsidePress={false}
      />,
    );

    fireEvent.mouseDown(document.body);

    expect(screen.getByRole("listbox")).toBeTruthy();
  });
});
