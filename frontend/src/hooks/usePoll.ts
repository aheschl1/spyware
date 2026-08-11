import { useEffect, useRef } from "react"

// Self-scheduling poll: runs immediately, then every `ms`; skips ticks while
// the tab is hidden and fires again the moment it becomes visible. The
// callback ref is refreshed each render so the poll always sees fresh state.
export function usePoll(fn: () => void | Promise<void>, ms: number): void {
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    let timer: number | undefined
    let stopped = false

    const tick = async () => {
      if (stopped) return
      if (!document.hidden) await fnRef.current()
      if (!stopped) timer = window.setTimeout(tick, ms)
    }
    void tick()

    const onVisibility = () => {
      if (!document.hidden && !stopped) {
        window.clearTimeout(timer)
        void tick()
      }
    }
    document.addEventListener("visibilitychange", onVisibility)
    return () => {
      stopped = true
      window.clearTimeout(timer)
      document.removeEventListener("visibilitychange", onVisibility)
    }
  }, [ms])
}
