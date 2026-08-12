// Time formatting shared by the timeline, player, and search results.

export function fmtClock(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m)
  return `${h > 0 ? `${h}:` : ""}${mm}:${String(s).padStart(2, "0")}`
}

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function fmtDuration(startIso: string, endIso: string | null): string {
  if (!endIso) return "live"
  return fmtClock(new Date(endIso).getTime() - new Date(startIso).getTime())
}

export function shortId(id: string): string {
  return id.slice(0, 8)
}
