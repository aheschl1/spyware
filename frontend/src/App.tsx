import { useEffect, useState } from "react"
import { clearToken, getToken, setExpiryHandler } from "./api/auth"
import { api, type MeRead } from "./api/client"
import Gate from "./components/Gate"
import SearchView from "./components/SearchView"
import SessionList from "./components/SessionList"
import SessionView from "./components/SessionView"
import SpeakersView from "./components/SpeakersView"

type Tab = "sessions" | "search" | "speakers"

type View =
  | { kind: "list"; tab: Tab }
  | { kind: "session"; id: string; seekMs?: number }

export default function App() {
  const [authed, setAuthed] = useState(() => getToken() !== null)
  const [view, setView] = useState<View>({ kind: "list", tab: "sessions" })
  const [me, setMe] = useState<MeRead | null>(null)

  useEffect(() => {
    setExpiryHandler(() => setAuthed(false))
  }, [])

  useEffect(() => {
    if (!authed) {
      setMe(null)
      return
    }
    void api.GET("/v1/me").then(({ data }) => {
      if (data) setMe(data)
    })
  }, [authed])

  if (!authed) return <Gate onAuthed={() => setAuthed(true)} />

  const openSession = (id: string, seekMs?: number) =>
    setView(seekMs === undefined ? { kind: "session", id } : { kind: "session", id, seekMs })
  const activeTab = view.kind === "list" ? view.tab : "sessions"

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">spyware</span>
        <nav className="tabs">
          {(["sessions", "search", "speakers"] as const).map((tab) => (
            <button
              key={tab}
              className={`tab ${activeTab === tab && view.kind === "list" ? "active" : ""}`}
              onClick={() => setView({ kind: "list", tab })}
            >
              {tab}
            </button>
          ))}
        </nav>
        <div className="topbar-right">
          {me && <span className="whoami">{me.user.display_name ?? me.user.email}</span>}
          <button
            className="btn ghost slim"
            onClick={() => {
              clearToken()
              setAuthed(false)
            }}
          >
            log out
          </button>
        </div>
      </header>

      <main className="main">
        {view.kind === "session" ? (
          <SessionView
            key={view.id}
            sessionId={view.id}
            seekMs={view.seekMs}
            onBack={() => setView({ kind: "list", tab: "sessions" })}
            onOpenSession={openSession}
          />
        ) : view.tab === "sessions" ? (
          <SessionList onOpen={openSession} />
        ) : view.tab === "search" ? (
          <SearchView onOpen={openSession} />
        ) : (
          <SpeakersView onOpen={openSession} />
        )}
      </main>
    </div>
  )
}
