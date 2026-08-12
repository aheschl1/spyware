import { useEffect, useState } from "react"

export type Playhead = {
  ms: number
  playing: boolean
  durationMs: number | null
}

const IDLE: Playhead = { ms: 0, playing: false, durationMs: null }

// Tracks an <audio> element's position. "smooth" rides requestAnimationFrame
// while playing (the strip's playhead glides); "coarse" quantizes to 250ms so
// components that only care about which utterance is current re-render rarely.
export function usePlayhead(
  audio: HTMLAudioElement | null,
  mode: "smooth" | "coarse" = "smooth",
): Playhead {
  const [head, setHead] = useState<Playhead>(IDLE)

  useEffect(() => {
    if (!audio) {
      setHead(IDLE)
      return
    }
    let raf = 0
    const read = () => {
      const rawMs = audio.currentTime * 1000
      const next: Playhead = {
        ms: mode === "coarse" ? Math.floor(rawMs / 250) * 250 : rawMs,
        playing: !audio.paused && !audio.ended,
        durationMs: Number.isFinite(audio.duration) ? audio.duration * 1000 : null,
      }
      // Same-reference bailout keeps coarse mode from re-rendering per tick.
      setHead((prev) =>
        prev.ms === next.ms &&
        prev.playing === next.playing &&
        prev.durationMs === next.durationMs
          ? prev
          : next,
      )
    }
    const tick = () => {
      read()
      if (!audio.paused && !audio.ended) raf = requestAnimationFrame(tick)
    }
    const start = () => {
      cancelAnimationFrame(raf)
      if (mode === "smooth") raf = requestAnimationFrame(tick)
      else read()
    }
    const events = ["play", "pause", "seeked", "timeupdate", "loadedmetadata", "ended"]
    audio.addEventListener("play", start)
    for (const name of events.slice(1)) audio.addEventListener(name, read)
    read()
    if (!audio.paused) start()
    return () => {
      cancelAnimationFrame(raf)
      audio.removeEventListener("play", start)
      for (const name of events.slice(1)) audio.removeEventListener(name, read)
    }
  }, [audio, mode])

  return head
}
