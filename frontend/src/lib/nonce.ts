// A fresh idempotency nonce for a workbench write (issues #59/#86). Uses
// crypto.getRandomValues — NOT crypto.randomUUID — deliberately: this is a
// self-hosted tool an operator may reach over plain HTTP on a LAN, where
// randomUUID (secure-context only) is undefined, while getRandomValues works
// everywhere. 16 bytes → a 32-char hex string, inside the routes' [8, 64] bound.
export function makeNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}
