import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Multi-entry static compiler (NO Astro, NO SSR): every entry becomes a
// content-hashed bundle plus a `.vite/manifest.json` the Python side reads to
// resolve entry -> hashed file. `vite build` has no server-runtime concept, so
// nothing here can violate the "no Node at operator runtime" invariant.
export default defineConfig({
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
