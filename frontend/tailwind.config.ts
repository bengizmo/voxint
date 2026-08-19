import type { Config } from "tailwindcss";

// darkMode: "media" (NOT "class") so islands follow the same OS light/dark
// signal base.html's `:root { color-scheme: light dark; }` already uses.
// Islands must not fight the page for a `dark` class.
export default {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      // Alias the base.html design tokens (issue #90) — never duplicate the
      // hex; CSS owns the values. RGB-channel form so opacity utilities keep
      // working: `bg-seg/20` -> rgb(var(--seg-rgb) / 0.2). Islands theme from
      // the same single source as the page chrome.
      colors: {
        seg: "rgb(var(--seg-rgb) / <alpha-value>)",
        splitword: "rgb(var(--splitword-rgb) / <alpha-value>)",
        line: "rgb(var(--line-rgb) / <alpha-value>)",
      },
    },
  },
  plugins: [],
} satisfies Config;
