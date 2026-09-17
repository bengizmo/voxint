// @vitest-environment jsdom

import { useLayoutEffect } from "react";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  APPEND_CHUNK,
  INITIAL_CHUNK,
  useProgressiveRender,
  type UseProgressiveRenderOptions,
} from "./progressive-render";

class MockIntersectionObserver {
  static instances: MockIntersectionObserver[] = [];
  observe = vi.fn();
  disconnect = vi.fn();

  constructor(
    private callback: IntersectionObserverCallback,
    readonly options: IntersectionObserverInit,
  ) {
    MockIntersectionObserver.instances.push(this);
  }

  trigger(isIntersecting = true) {
    this.callback(
      [{ isIntersecting } as IntersectionObserverEntry],
      this as unknown as IntersectionObserver,
    );
  }
}

function latestObserver() {
  return MockIntersectionObserver.instances[
    MockIntersectionObserver.instances.length - 1
  ];
}

function mountHook(initialProps: UseProgressiveRenderOptions) {
  return renderHook(
    (options) => {
      const result = useProgressiveRender(options);
      const ref = result.sentinelRef;
      // Attach a real sentinel during commit, before the observer's effect.
      useLayoutEffect(() => {
        const sentinel = document.createElement("div");
        document.body.append(sentinel);
        ref.current = sentinel;
        return () => {
          ref.current = null;
          sentinel.remove();
        };
      }, [ref]);
      return result;
    },
    { initialProps },
  );
}

beforeEach(() => {
  MockIntersectionObserver.instances = [];
  vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("useProgressiveRender", () => {
  it("starts with the initial chunk for a large transcript", () => {
    const { result } = mountHook({ totalCount: 1000 });
    expect(result.current.renderedEnd).toBe(INITIAL_CHUNK);
    expect(latestObserver().observe).toHaveBeenCalledWith(
      result.current.sentinelRef.current,
    );
    expect(latestObserver().options).toEqual({
      root: null,
      rootMargin: "0px 0px 600px 0px",
    });
  });

  it.each([0, 50, INITIAL_CHUNK])(
    "renders all %i rows of a small transcript",
    (totalCount) => {
      const { result } = mountHook({ totalCount });
      expect(result.current.renderedEnd).toBe(totalCount);
      expect(MockIntersectionObserver.instances).toHaveLength(0);
    },
  );

  it("appends a chunk on intersection and re-observes after commit", () => {
    const { result } = mountHook({ totalCount: 1000 });
    const first = latestObserver();
    act(() => first.trigger(false));
    expect(result.current.renderedEnd).toBe(INITIAL_CHUNK);
    act(() => {
      first.trigger();
      first.trigger();
    });
    expect(result.current.renderedEnd).toBe(INITIAL_CHUNK + APPEND_CHUNK);
    expect(first.disconnect).toHaveBeenCalledOnce();
    act(() => latestObserver().trigger());
    expect(result.current.renderedEnd).toBe(INITIAL_CHUNK + 2 * APPEND_CHUNK);
  });

  it("caps observer appends at the total and ignores obsolete callbacks", () => {
    const { result } = mountHook({ totalCount: 250 });
    const observer = latestObserver();
    act(() => observer.trigger());
    expect(result.current.renderedEnd).toBe(250);
    expect(observer.disconnect).toHaveBeenCalledOnce();
    act(() => observer.trigger());
    expect(result.current.renderedEnd).toBe(250);
  });

  it("expands to the requested target plus a chunk, capped at the total", () => {
    const { result } = mountHook({ totalCount: 1000 });
    act(() => result.current.ensureRendered(500));
    expect(result.current.renderedEnd).toBe(500 + APPEND_CHUNK);
    act(() => result.current.ensureRendered(999));
    expect(result.current.renderedEnd).toBe(1000);
  });

  it("does nothing for a target already rendered, including the last row", () => {
    const { result } = mountHook({ totalCount: 1000 });
    act(() => result.current.ensureRendered(INITIAL_CHUNK - 1));
    expect(result.current.renderedEnd).toBe(INITIAL_CHUNK);
  });

  it("shows all rows", () => {
    const { result } = mountHook({ totalCount: 1000 });
    act(() => result.current.showAll());
    expect(result.current.renderedEnd).toBe(1000);
  });

  it("clamps after removal without expanding when rows are added", () => {
    const { result, rerender } = mountHook({ totalCount: 1000 });
    rerender({ totalCount: 120 });
    expect(result.current.renderedEnd).toBe(120);
    rerender({ totalCount: 1500 });
    expect(result.current.renderedEnd).toBe(120);
    act(() => latestObserver().trigger());
    expect(result.current.renderedEnd).toBe(220);
  });

  it("includes a distant initial minimum index", () => {
    const { result } = mountHook({ totalCount: 1000, minIndex: 500 });
    expect(result.current.renderedEnd).toBe(500 + APPEND_CHUNK);
  });

  it("expands for a minimum at the exclusive boundary and preserves growth", () => {
    const { result, rerender } = mountHook({ totalCount: 1000 });
    rerender({ totalCount: 1000, minIndex: INITIAL_CHUNK });
    expect(result.current.renderedEnd).toBe(INITIAL_CHUNK + APPEND_CHUNK);
    rerender({ totalCount: 1200, minIndex: 1100 });
    expect(result.current.renderedEnd).toBe(1200);
    rerender({ totalCount: 1200, minIndex: 0 });
    expect(result.current.renderedEnd).toBe(1200);
  });

  it("disconnects on unmount", () => {
    const { unmount } = mountHook({ totalCount: 1000 });
    const observer = latestObserver();
    unmount();
    expect(observer.disconnect).toHaveBeenCalledOnce();
  });
});
