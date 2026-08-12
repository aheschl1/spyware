import { useCallback, useState } from "react"
import { api, type AbResultsRead, type SessionRead } from "../api/client"
import { fmtDate, fmtDuration, shortId } from "../format"
import { usePoll } from "../hooks/usePoll"

const POLL_MS = 10_000

export default function AbOverview({ onVote }: { onVote: (sessionId: string) => void }) {
  const [results, setResults] = useState<AbResultsRead | null>(null)
  const [sessions, setSessions] = useState<SessionRead[] | null>(null)
  const [queued, setQueued] = useState<Record<string, boolean>>({})

  const refresh = useCallback(async () => {
    const [tally, list] = await Promise.all([
      api.GET("/v1/ab/results"),
      api.GET("/v1/sessions", { params: { query: { limit: 200, offset: 0 } } }),
    ])
    if (tally.data) setResults(tally.data)
    if (list.data) setSessions(list.data.items)
  }, [])
  usePoll(refresh, POLL_MS)

  const enroll = async (sessionId: string) => {
    setQueued((prev) => ({ ...prev, [sessionId]: true }))
    const { data } = await api.POST("/v1/sessions/{session_id}/ab", {
      params: { path: { session_id: sessionId } },
    })
    if (!data) setQueued((prev) => ({ ...prev, [sessionId]: false }))
  }

  const votesFor = (sessionId: string) =>
    results?.sessions.find((s) => s.session_id === sessionId)?.votes ?? 0

  return (
    <div className="ab-overview">
      <div className="ab-tally">
        <div className="row-title">model × strategy wins</div>
        {results === null ? (
          <div className="skeleton" style={{ height: 60 }} />
        ) : results.total === 0 ? (
          <div className="empty">No votes yet — enroll a session and start voting.</div>
        ) : (
          <table className="ab-tally-table">
            <thead>
              <tr>
                <th>model</th>
                <th>strategy</th>
                <th>wins</th>
                <th>share</th>
              </tr>
            </thead>
            <tbody>
              {results.tally.map((row) => (
                <tr key={`${row.model}:${row.strategy}`}>
                  <td>{row.model}</td>
                  <td>{row.strategy}</td>
                  <td>{row.wins}</td>
                  <td>
                    <span className="ab-share">
                      <span
                        className="ab-share-fill"
                        style={{ width: `${(row.wins / results.total) * 100}%` }}
                      />
                    </span>
                    {Math.round((row.wins / results.total) * 100)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {results !== null && results.total > 0 && (
          <div className="row-dim">{results.total} vote(s) total</div>
        )}
      </div>

      <div className="list">
        {sessions === null ? (
          [64, 64, 64].map((height, i) => <div key={i} className="skeleton" style={{ height }} />)
        ) : sessions.length === 0 ? (
          <div className="empty">No sessions yet.</div>
        ) : (
          sessions.map((session) => (
            <div key={session.id} className="row ab-session-row">
              <div className="row-main">
                <span className="row-title">{session.label ?? shortId(session.id)}</span>
                <span className="row-sub">
                  {fmtDate(session.started_at)} · {fmtDuration(session.started_at, session.ended_at)}
                </span>
              </div>
              {votesFor(session.id) > 0 && (
                <span className="chip">{votesFor(session.id)} voted</span>
              )}
              <button
                className="btn ghost slim"
                disabled={queued[session.id]}
                onClick={() => void enroll(session.id)}
              >
                {queued[session.id] ? "queued" : "generate"}
              </button>
              <button className="btn primary slim" onClick={() => onVote(session.id)}>
                vote
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
