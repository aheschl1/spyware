import { useCallback, useEffect, useRef, useState } from "react"
import { api, type TimelineEvent } from "../api/client"

const PAGE = 200
// The visual strip wants the whole session; this cap only guards against a
// pathological event count, and the hook reports when it was hit.
const MAX_EVENTS = 5000

export type SessionEvents = {
  events: TimelineEvent[] | null // null until the first page lands
  complete: boolean
  truncated: boolean
}

// Sweeps the whole paginated timeline into memory, updating progressively so
// the strip draws as pages land. `refreshKey` bumps force a refetch (after a
// speaker was labeled or merged, names must propagate).
export function useSessionEvents(sessionId: string, refreshKey: number): SessionEvents {
  const [events, setEvents] = useState<TimelineEvent[] | null>(null)
  const [complete, setComplete] = useState(false)
  const [truncated, setTruncated] = useState(false)
  const generation = useRef(0)

  const load = useCallback(async () => {
    const gen = ++generation.current
    setComplete(false)
    setTruncated(false)
    const all: TimelineEvent[] = []
    for (;;) {
      const { data } = await api.GET("/v1/sessions/{session_id}/timeline", {
        params: {
          path: { session_id: sessionId },
          query: { limit: PAGE, offset: all.length },
        },
      })
      if (generation.current !== gen) return
      if (!data) break
      all.push(...data.items)
      setEvents([...all])
      if (!data.has_more) break
      if (all.length >= MAX_EVENTS) {
        setTruncated(true)
        break
      }
    }
    if (generation.current === gen) {
      setEvents([...all])
      setComplete(true)
    }
  }, [sessionId])

  useEffect(() => {
    void load()
  }, [load, refreshKey])

  return { events, complete, truncated }
}
