import type { Config } from "tailwindcss";

// darkMode: "media" is inert in practice: islands use ZERO `dark:` utilities
// and theme entirely through base.html's CSS variables, which since #94
// re-resolve under both the guarded `prefers-color-scheme: dark` block and
// the explicit `:root[data-theme="dark"]` block. Islands must not fight the
// page for a `dark` class.
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
      fontFamily: {
        ui: "var(--font-ui)",
        mono: "var(--font-mono)",
      },
    },
  },
  plugins: [],
} satisfies Config;
