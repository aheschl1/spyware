// Token lifecycle. Login tokens carry no client-visible expiry (the server
// enforces its own), so there is no local expiry check — a 401 anywhere is
// the signal, routed through the expiry handler so App can flip to the Gate.

const KEY = "audiopipeline.token"

export function getToken(): string | null {
  return localStorage.getItem(KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(KEY, token)
}

// Local-only: the API has no logout route; the server-side row stays live.
export function clearToken(): void {
  localStorage.removeItem(KEY)
}

let onExpire: (() => void) | null = null

export function setExpiryHandler(fn: () => void): void {
  onExpire = fn
}

export function expire(): void {
  clearToken()
  onExpire?.()
}
