// ESLint 9 flat config (the legacy .eslintrc.cjs format is not read by default
// under ESLint 9). TypeScript + React-hooks rules; `npm run lint` runs with
// --max-warnings=0 so any warning fails CI.
import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
    },
  },
);
