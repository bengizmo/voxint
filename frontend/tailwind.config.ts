import type { Config } from "tailwindcss";

// darkMode: "media" (NOT "class") so islands follow the same OS light/dark
// signal base.html's `:root { color-scheme: light dark; }` already uses.
// Islands must not fight the page for a `dark` class.
export default {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
