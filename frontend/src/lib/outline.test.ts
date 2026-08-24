import { describe, expect, it } from "vitest";

import { formatClock, totalDropped } from "./outline";

describe("formatClock", () => {
  it("renders mm:ss under an hour", () => {
    expect(formatClock(0)).toBe("0:00");
    expect(formatClock(5)).toBe("0:05");
    expect(formatClock(65)).toBe("1:05");
    expect(formatClock(600)).toBe("10:00");
  });

  it("renders h:mm:ss at and past an hour", () => {
    expect(formatClock(3600)).toBe("1:00:00");
    expect(formatClock(3661)).toBe("1:01:01");
  });

  it("truncates sub-second and floors negatives/NaN to zero", () => {
    expect(formatClock(43.9)).toBe("0:43");
    expect(formatClock(-5)).toBe("0:00");
    expect(formatClock(Number.NaN)).toBe("0:00");
  });
});

describe("totalDropped", () => {
  it("sums the three drop counters", () => {
    expect(
      totalDropped({ droppedUnlocatable: 1, droppedOutOfRun: 2, droppedUnresolved: 3 }),
    ).toBe(6);
    expect(
      totalDropped({ droppedUnlocatable: 0, droppedOutOfRun: 0, droppedUnresolved: 0 }),
    ).toBe(0);
  });
});
