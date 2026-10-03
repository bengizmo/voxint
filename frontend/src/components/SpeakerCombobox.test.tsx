// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
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

// [#772] A failed create must hand focus back to the input. Browsers drop
// focus from an element when it becomes disabled; jsdom neither does that nor
// lets a disabled element blur, so the tests blur the input just before the
// create starts to reproduce where the browser leaves focus.
describe("SpeakerCombobox focus after a failed create", () => {
  function deferred() {
    let resolve!: (ok: boolean) => void;
    const promise = new Promise<boolean>((r) => {
      resolve = r;
    });
    return { promise, resolve };
  }

  it("returns focus to the input so Escape closes the list", async () => {
    const pending = deferred();
    const onCreate = vi.fn(() => pending.promise);
    render(
      <SpeakerCombobox
        speakers={speakers}
        mode="command"
        label="S2: choose who this is"
        onSelect={vi.fn()}
        onCreate={onCreate}
        autoFocus
      />,
    );
    const input = screen.getByRole("combobox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Mara" } });
    input.blur();
    fireEvent.click(screen.getByRole("option", { name: 'Create "Mara"' }));

    expect(onCreate).toHaveBeenCalledWith("Mara");
    expect(input.disabled).toBe(true);
    expect(document.activeElement).toBe(document.body);

    await act(async () => {
      pending.resolve(false);
      await pending.promise;
    });

    expect(input.disabled).toBe(false);
    expect(input.value).toBe("Mara");
    expect(document.activeElement).toBe(input);

    fireEvent.keyDown(input, { key: "Escape" });
    expect(screen.queryByRole("listbox")).toBeNull();
  });

  it("waits for the host to re-enable the picker before refocusing", async () => {
    const pending = deferred();
    const props = {
      speakers,
      mode: "command" as const,
      label: "S2: choose who this is",
      onSelect: vi.fn(),
      onCreate: vi.fn(() => pending.promise),
      autoFocus: true,
    };
    const { rerender } = render(<SpeakerCombobox {...props} />);
    const input = screen.getByRole("combobox") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "Mara" } });
    input.blur();
    fireEvent.keyDown(input, { key: "Enter" });

    // The rail marks itself busy for the length of the enroll request.
    rerender(<SpeakerCombobox {...props} disabled />);
    await act(async () => {
      pending.resolve(false);
      await pending.promise;
    });
    expect(input.disabled).toBe(true);
    expect(document.activeElement).toBe(document.body);

    rerender(<SpeakerCombobox {...props} disabled={false} />);
    expect(document.activeElement).toBe(input);
  });

  it("does not refocus the input after a successful create", async () => {
    const onCreate = vi.fn(async () => true);
    render(
      <SpeakerCombobox
        speakers={speakers}
        mode="command"
        label="S2: choose who this is"
        onSelect={vi.fn()}
        onCreate={onCreate}
        autoFocus
      />,
    );
    const input = screen.getByRole("combobox");
    fireEvent.change(input, { target: { value: "Mara" } });
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter" });
    });

    expect(screen.queryByRole("listbox")).toBeNull();
    expect(screen.queryByRole("combobox")).toBeNull();
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "S2: choose who this is" }),
    );
  });
});
