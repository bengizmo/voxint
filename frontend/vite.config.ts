import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Multi-entry static compiler (NO Astro, NO SSR): every entry becomes a
// content-hashed bundle plus a `.vite/manifest.json` the Python side reads to
// resolve entry -> hashed file. `vite build` has no server-runtime concept, so
// nothing here can violate the "no Node at operator runtime" invariant.
export default defineConfig({
  // Assets are served ONLY under /static/app/ (the auth-aware app_asset route),
  // so the base must match: it sets the root the modulepreload helper and any
  // CSS/chunk dependency URLs resolve against. Left at the default "/", a future
  // island that shares a code-split chunk would emit <link modulepreload
  // href="/assets/..."> (404, silent hydration failure). Manifest `file` values
  // stay relative regardless, so app.py's asset_url (which prepends /static/app/)
  // is unaffected. See issue #48 review.
  base: "/static/app/",
  plugins: [react()],
  build: {
    manifest: true, // emits dist/.vite/manifest.json
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: "src/main.ts",
        tailwind: "src/styles/tailwind.css",
        "transcript-player": "src/entries/transcript-player.tsx",
      },
    },
  },
});
