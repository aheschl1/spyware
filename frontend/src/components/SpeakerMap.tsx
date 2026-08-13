import { useEffect, useMemo, useState } from "react"
import type { ComponentType } from "react"
import { api, type ProjectionPointRead, type SpeakerProjectionRead } from "../api/client"
import { shortId } from "../format"
import { hue } from "../speakers"
import SpeakerMapInspector from "./SpeakerMapInspector"

const AXES = [
  { key: "12", label: "PC1 · PC2", x: 0, y: 1 },
  { key: "13", label: "PC1 · PC3", x: 0, y: 2 },
  { key: "23", label: "PC2 · PC3", x: 1, y: 2 },
] as const

const UNASSIGNED = "unassigned"
const LEGEND_LIMIT = 14

// Plotly is ~450kB gzipped — far larger than the rest of the app. Loading it
// through a dynamic import keeps it in its own chunk, fetched only when this
// tab is opened.
let plotPromise: Promise<ComponentType<Record<string, unknown>>> | null = null

function loadPlot() {
  if (!plotPromise) {
    plotPromise = Promise.all([
      import("plotly.js-gl2d-dist-min"),
      import("react-plotly.js/factory"),
    ]).then(
      ([plotly, factory]) =>
        factory.default(plotly.default) as ComponentType<Record<string, unknown>>,
    )
  }
  return plotPromise
}

function markerSize(talkMs: number | null | undefined): number {
  if (!talkMs) return 6
  return Math.min(14, Math.max(5, Math.sqrt(talkMs / 1000) * 2.2))
}

export default function SpeakerMap({
  onOpen,
}: {
  onOpen: (sessionId: string, seekMs?: number) => void
}) {
  const [Plot, setPlot] = useState<ComponentType<Record<string, unknown>> | null>(null)
  const [data, setData] = useState<SpeakerProjectionRead | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [model, setModel] = useState<string | null>(null)
  const [includeUnassigned, setIncludeUnassigned] = useState(true)
  const [axes, setAxes] = useState<(typeof AXES)[number]>(AXES[0])
  const [selected, setSelected] = useState<ProjectionPointRead | null>(null)

  useEffect(() => {
    void loadPlot().then(setPlot, () => setError("could not load the plotting library"))
  }, [])

  useEffect(() => {
    let live = true
    setError(null)
    void api
      .GET("/v1/speakers/projection", {
        params: {
          query: {
            include_unassigned: includeUnassigned,
            ...(model ? { model } : {}),
          },
        },
      })
      .then(({ data: body, error: failed }) => {
        if (!live) return
        if (failed || !body) setError("could not load the projection")
        else setData(body)
      })
  }, [model, includeUnassigned])

  const byId = useMemo(() => {
    const index = new Map<string, ProjectionPointRead>()
    for (const point of data?.points ?? []) index.set(point.artifact_id, point)
    return index
  }, [data])

  // One trace per cluster: that is what makes Plotly's legend isolate voices
  // for free, and it keeps colour keyed the way the timeline keys it.
  const traces = useMemo(() => {
    if (!data) return []
    const groups = new Map<string, ProjectionPointRead[]>()
    for (const point of data.points) {
      const key = point.speaker_id ?? UNASSIGNED
      const bucket = groups.get(key)
      if (bucket) bucket.push(point)
      else groups.set(key, [point])
    }

    return [...groups.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .map(([key, points], index) => {
        const unassigned = key === UNASSIGNED
        const first = points[0]
        const name = unassigned
          ? `unassigned (${points.length})`
          : `${first?.name ?? shortId(key)} (${points.length})`
        return {
          type: "scattergl",
          mode: "markers",
          name,
          // A corpus is mostly one- and two-print clusters; listing all of
          // them buries the voices that matter and covers the plot. The tail
          // still draws in its own colour, and the inspector still names it.
          showlegend: index < LEGEND_LIMIT,
          x: points.map((point) => point.coords[axes.x]),
          y: points.map((point) => point.coords[axes.y]),
          customdata: points.map((point) => [
            point.artifact_id,
            point.session_label ?? shortId(point.session_id),
            point.talk_ms ? (point.talk_ms / 1000).toFixed(1) : "?",
            point.distance === null || point.distance === undefined
              ? "—"
              : point.distance.toFixed(3),
          ]),
          marker: {
            size: points.map((point) => markerSize(point.talk_ms)),
            color: unassigned ? "rgba(141,150,168,.45)" : `hsl(${hue(key)} 55% 55%)`,
            line: {
              width: unassigned ? 1 : 0.5,
              color: unassigned ? "#8d96a8" : "rgba(0,0,0,.45)",
            },
          },
          hovertemplate:
            `<b>${name}</b><br>%{customdata[1]}<br>` +
            "speech %{customdata[2]}s · distance %{customdata[3]}<extra></extra>",
        }
      })
  }, [data, axes])

  const layout = useMemo(
    () => ({
      // Plotly keeps the user's zoom/pan while uirevision is unchanged. The
      // basis id moves exactly when the axes stop meaning the same thing.
      uirevision: data?.basis_id ?? "empty",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#0b0d12",
      font: { color: "#8d96a8", size: 11 },
      margin: { l: 48, r: 176, t: 12, b: 40 },
      dragmode: "pan",
      hovermode: "closest",
      showlegend: true,
      // Outside the plotting area: 400 points must not hide behind a legend.
      legend: {
        x: 1.015,
        xanchor: "left",
        y: 1,
        yanchor: "top",
        bgcolor: "rgba(0,0,0,0)",
        title: { text: `largest voices (${LEGEND_LIMIT})`, font: { size: 10 } },
      },
      xaxis: {
        title: { text: `PC${axes.x + 1}` },
        gridcolor: "#1f2530",
        zerolinecolor: "#272d38",
      },
      yaxis: {
        title: { text: `PC${axes.y + 1}` },
        gridcolor: "#1f2530",
        zerolinecolor: "#272d38",
        scaleanchor: "x",
      },
    }),
    [data?.basis_id, axes],
  )

  if (error) return <div className="banner">{error}</div>
  if (!data || !Plot) return <div className="loading">loading the voice map…</div>
  if (!data.model) return <div className="empty">no voice-prints yet</div>

  const [pc1, pc2] = [data.explained_variance_ratio[0], data.explained_variance_ratio[1]]
  const shown = Math.round(((pc1 ?? 0) + (pc2 ?? 0)) * 100)

  return (
    <div className="map-view">
      <div className="map-toolbar">
        <div className="mode-toggle">
          {AXES.map((pair) => (
            <button
              key={pair.key}
              className={`mode ${pair.key === axes.key ? "active" : ""}`}
              onClick={() => setAxes(pair)}
            >
              {pair.label}
            </button>
          ))}
        </div>

        {data.available_models.length > 1 && (
          <select
            className="input slim"
            value={model ?? data.model}
            onChange={(event) => setModel(event.target.value)}
          >
            {data.available_models.map((entry) => (
              <option key={entry.model} value={entry.model}>
                {entry.model} ({entry.embeddings})
              </option>
            ))}
          </select>
        )}

        <label className="map-check">
          <input
            type="checkbox"
            checked={includeUnassigned}
            onChange={(event) => setIncludeUnassigned(event.target.checked)}
          />
          unassigned
        </label>

        <span className="row-dim">
          {data.returned} of {data.fit_points} voice-prints
          {data.truncated && " (sampled)"}
        </span>
      </div>

      <p className="map-caveat">
        PC1+PC2 explain <strong>{shown}%</strong> of the variance across{" "}
        {data.fit_points} voice-prints — voices that look close here may not be.
        Click a point for its true distance.
      </p>

      <div className="map-body">
        <div className="map-plot">
          <Plot
            data={traces}
            layout={layout}
            config={{ displaylogo: false, scrollZoom: true, responsive: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
            onClick={(event: { points?: { customdata?: unknown[] }[] }) => {
              const id = event.points?.[0]?.customdata?.[0]
              if (typeof id === "string") setSelected(byId.get(id) ?? null)
            }}
          />
        </div>

        {selected && (
          <SpeakerMapInspector
            point={selected}
            onOpen={onOpen}
            onClose={() => setSelected(null)}
          />
        )}
      </div>
    </div>
  )
}
