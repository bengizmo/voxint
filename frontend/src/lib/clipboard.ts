// Write text to the clipboard, returning whether it actually succeeded (issue #86).
//
// `navigator.clipboard` is undefined on a non-secure LAN context (plain http),
// exactly where a self-hosted operator often runs, and `writeText` can also reject
// (permissions, focus). Both paths resolve to `false` so every caller can fall back
// to an honest manual-copy affordance rather than claiming a success it did not
// achieve. Extracted from ReviewStepper's #83 "Copy raw text" handler so the raw-copy
// and the annotation pull-quote copy share one path.
export async function writeClipboard(text: string): Promise<boolean> {
  try {
    if (!navigator.clipboard?.writeText) return false;
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
