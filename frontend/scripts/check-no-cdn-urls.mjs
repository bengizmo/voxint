// Offline self-host guarantee, run against the EXACT bytes that get COPY'd into
// the image. Fails the build if any built JS/CSS references an external host
// (Google Fonts, unpkg, jsdelivr, esm.sh, cdnjs, ...). Allows loopback hosts
// (localhost / 127.0.0.1 / 0.0.0.0). Node-only, never a pytest — the Python
// suite stays Node-free.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const DIST = new URL("../dist/", import.meta.url).pathname;
const SCANNED_EXT = new Set([".js", ".mjs", ".css", ".map"]);
// Hosts that are NOT runtime asset loads and so cannot violate offline
// self-host. Each is a string constant embedded by a dependency, never fetched:
//   - loopback: same-origin dev references.
//   - www.w3.org: XML/SVG/XHTML *namespace URIs* (e.g. the SVG namespace passed
//     to document.createElementNS) — DOM-spec identifiers, mandated constants,
//     never network requests. React DOM emits these for every app.
//   - react.dev: React's dev-mode error/warning messages carry documentation
//     links as plain strings ("visit https://react.dev/link/..."); they are
//     console text, never loaded.
// Real CDN asset hosts (unpkg, jsdelivr, esm.sh, cdnjs, fonts.googleapis.com,
// ...) are deliberately NOT allowlisted and still fail the build.
const ALLOWED_HOSTS = new Set([
  "localhost",
  "127.0.0.1",
  "0.0.0.0",
  "www.w3.org",
  "react.dev",
]);

// Matches http(s) and protocol-relative URLs, capturing the host.
const URL_RE = /(?:https?:)?\/\/([a-z0-9.-]+)(?::\d+)?/gi;

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
  console.error(`check-no-cdn-urls: cannot read dist/ (did vite build run?): ${err.message}`);
  process.exit(1);
}

const violations = [];
for (const file of files) {
  const ext = file.slice(file.lastIndexOf("."));
  if (!SCANNED_EXT.has(ext)) continue;
  const text = readFileSync(file, "utf8");
  for (const match of text.matchAll(URL_RE)) {
    const host = match[1].toLowerCase();
    if (!ALLOWED_HOSTS.has(host)) {
      violations.push({ file, host, snippet: match[0] });
    }
  }
}

if (violations.length > 0) {
  console.error("check-no-cdn-urls: external host reference(s) found in built assets:");
  for (const v of violations) {
    console.error(`  ${v.file}: ${v.snippet}`);
  }
  console.error(
    "Offline self-host is a hard invariant: no runtime CDN/network calls in built assets.",
  );
  process.exit(1);
}

console.log(`check-no-cdn-urls: OK (${files.length} files scanned, no external hosts).`);
