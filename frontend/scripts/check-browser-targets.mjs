// Browser-floor guard, run against the built bytes (#655). vite.config.ts pins
// Vite 6's build targets; this fails the build if the output still uses syntax
// those browsers can't read, so a toolchain change can't drop them silently.
//   - CSS: range media queries such as `(width>=640px)`. Safari before 16.4
//     ignores the whole rule, so every responsive breakpoint stops applying.
//   - JS: logical assignment (`??=`, `||=`, `&&=`, ES2021). The source uses it,
//     so it shows up as soon as the JS target stops being honored.
// Node-only, never a pytest: the Python suite stays Node-free.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const DIST = new URL("../dist/", import.meta.url).pathname;

// `(width>=640px)`, `(400px<width)`, `(height <= 2px)`: a comparison operator
// next to a media feature name, inside an @media prelude.
const RANGE_MEDIA_RE =
  /@media[^{]*\(\s*(?:(?:[a-z-]+)\s*[<>]=?|[^():]*[<>]=?\s*(?:width|height|aspect-ratio|resolution)\b)/i;
const LOGICAL_ASSIGN_RE = /\?\?=|\|\|=|&&=/;

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      out.push(...walk(full));
    } else {
      out.push(full);
    }
  }
  return out;
}

let files;
try {
  files = walk(DIST);
} catch (err) {
  console.error(
    `check-browser-targets: cannot read dist/ (did vite build run?): ${err.message}`,
  );
  process.exit(1);
}

const violations = [];
let scanned = 0;
for (const file of files) {
  const ext = file.slice(file.lastIndexOf("."));
  if (ext === ".css") {
    scanned += 1;
    const match = readFileSync(file, "utf8").match(RANGE_MEDIA_RE);
    if (match)
      violations.push({ file, what: "range media query", snippet: match[0] });
  } else if (ext === ".js" || ext === ".mjs") {
    scanned += 1;
    const match = readFileSync(file, "utf8").match(LOGICAL_ASSIGN_RE);
    if (match)
      violations.push({
        file,
        what: "ES2021 logical assignment",
        snippet: match[0],
      });
  }
}

if (violations.length > 0) {
  console.error(
    "check-browser-targets: built assets use syntax below the pinned browser floor:",
  );
  for (const v of violations) {
    console.error(`  ${v.file}: ${v.what}: ${v.snippet}`);
  }
  console.error("Check build.target / build.cssTarget in vite.config.ts.");
  process.exit(1);
}

console.log(`check-browser-targets: OK (${scanned} files scanned).`);
