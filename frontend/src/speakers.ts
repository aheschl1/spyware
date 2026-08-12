// Speaker presentation helpers shared by the speakers page, the session
// timeline, and the visual strip — one place so a cluster wears the same
// colour and label everywhere.

import { shortId } from "./format"

export function initials(name: string | null | undefined, id: string): string {
  if (!name) return id.slice(0, 2)
  const parts = name.trim().split(/\s+/)
  return ((parts[0]?.[0] ?? "") + (parts[1]?.[0] ?? "")).toUpperCase() || id.slice(0, 2)
}

// Deterministic accent per speaker so the same cluster always wears the
// same colour, labeled or not. Unclustered voices hash their local label.
export function hue(id: string): number {
  let h = 0
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) % 360
  return h
}

export function speakerLabel(speaker: { name?: string | null; id: string }): string {
  return speaker.name ?? shortId(speaker.id)
}

// Block-local diarization labels are namespaced (`b109176:SPEAKER_00`);
// only the tail is human-meaningful.
export function localLabelTail(label: string): string {
  return label.split(":").pop() ?? label
}
