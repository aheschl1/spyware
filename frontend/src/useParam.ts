import { useCallback } from "react"
import { useSearchParams } from "react-router"

/** A query-string value with a default; setting the default deletes the key. */
export function useSearchParam(key: string, fallback: string) {
  const [params, setParams] = useSearchParams()
  const value = params.get(key) ?? fallback
  const set = useCallback(
    (next: string | null, extra?: Record<string, string | null>) =>
      setParams(
        (prev) => {
          const out = new URLSearchParams(prev)
          const apply = (k: string, v: string | null) =>
            v === null || (k === key && v === fallback) ? out.delete(k) : out.set(k, v)
          apply(key, next)
          for (const [k, v] of Object.entries(extra ?? {})) apply(k, v)
          return out
        },
        { replace: true },
      ),
    [key, fallback, setParams],
  )
  return [value, set] as const
}
