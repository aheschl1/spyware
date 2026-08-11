import { useState, type FormEvent } from "react"
import { api, type AudioSearchRead } from "../api/client"
import { fmtClock, shortId } from "../format"

// Text->audio search over the CLAP window embeddings. Distances are only
// comparable within one query, so results show rank order, not a match
// percentage. A 502 means the embedding sidecar (audio-tagger) is down.
export default function SearchView({
  onOpen,
}: {
  onOpen: (id: string, seekMs?: number) => void
}) {
  const [q, setQ] = useState("")
  const [results, setResults] = useState<AudioSearchRead[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!q.trim()) return
    setBusy(true)
    setError(null)
    const { data, response } = await api.GET("/v1/search/audio", {
      params: { query: { q: q.trim(), limit: 30 } },
    })
    setBusy(false)
    if (data) {
      setResults(data.items)
    } else {
      setResults(null)
      setError(
        response.status === 502
          ? "the audio search service is offline"
          : "search failed",
      )
    }
  }

  return (
    <div className="search-view">
      <form className="search-bar" onSubmit={submit}>
        <input
          className="input"
          placeholder="describe a sound — “keyboard typing”, “a dog barking”, “music”…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          autoFocus
        />
        <button className="btn primary" disabled={busy || !q.trim()}>
          {busy ? "searching…" : "search"}
        </button>
      </form>

      {error && <div className="banner error">{error}</div>}
      {results && results.length === 0 && <div className="empty">No matching audio.</div>}
      {results && results.length > 0 && (
        <div className="list">
          {results.map((hit, rank) => (
            <button
              key={hit.artifact_id}
              className="row"
              onClick={() => onOpen(hit.session_id, hit.start_ms)}
            >
              <span className="rank">#{rank + 1}</span>
              <div className="row-main">
                <span className="row-title">
                  {fmtClock(hit.start_ms)} – {fmtClock(hit.end_ms)}
                  <span className="row-dim"> in {shortId(hit.session_id)}</span>
                </span>
                {hit.labels.length > 0 && (
                  <span className="chips">
                    {hit.labels.slice(0, 4).map((tag) => (
                      <span key={tag.label} className="chip">
                        {tag.label}
                      </span>
                    ))}
                  </span>
                )}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
