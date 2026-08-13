import { useEffect, useMemo, useState } from "react"
import type { ComponentType } from "react"
import { api, type ProjectionPointRead, type SpeakerProjectionRead } from "../api/client"
import { shortId } from "../format"
import { hue } from "../speakers"
import SpeakerMapInspector from "./SpeakerMapInspector"

// Each view names the components it shows, so the explained-variance figure
// is derived from the axes on screen rather than assumed to be PC1+PC2.
const VIEWS = [
  { key: "12", label: "PC1 · PC2", axes: [0, 1] },
  { key: "13", label: "PC1 · PC3", axes: [0, 2] },
  { key: "23", label: "PC2 · PC3", axes: [1, 2] },
  { key: "3d", label: "3D", axes: [0, 1, 2] },
] as const

type View = (typeof VIEWS)[number]

const UNASSIGNED = "unassigned"
const LEGEND_LIMIT = 14
const MUTED = "#8d96a8"
const GRID = "#1f2530"

let plotPromise: Promise<ComponentType<Record<string, unknown>>> | null = null

// Plotly dwarfs the rest of the app; a dynamic import keeps it in its own
// chunk, fetched only when this tab is opened.
function loadPlot() {
  if (!plotPromise) {
    plotPromise = Promise.all([
      import("../plotlyBundle"),
      import("react-plotly.js/factory"),
    ]).then(
      ([plotly, factory]) =>
        factory.default(plotly.default) as ComponentType<Record<string, unknown>>,
    )
  }
  return plotPromise
}

function markerSize(talkMs: number | null | undefined, solid: boolean): number {
  const base = talkMs ? Math.sqrt(talkMs / 1000) * 2.2 : 6
  // scatter3d renders markers larger for the same value than scattergl.
  return Math.min(solid ? 9 : 14, Math.max(solid ? 3 : 5, solid ? base * 0.6 : base))
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
  const [view, setView] = useState<View>(VIEWS[0])
  const [selected, setSelected] = useState<ProjectionPointRead | null>(null)

  const is3d = view.key === "3d"

  useEffect(() => {
    void loadPlot().then(setPlot, (reason) => {
      console.error("plotly load failed", reason)
      setError("could not load the plotting library")
    })
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
    return () => {
      live = false
    }
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

    const [ax, ay, az] = view.axes
    return [...groups.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .map(([key, points], index) => {
        const unassigned = key === UNASSIGNED
        const first = points[0]
        const name = unassigned
          ? `unassigned (${points.length})`
          : `${first?.name ?? shortId(key)} (${points.length})`
        return {
          type: is3d ? "scatter3d" : "scattergl",
          mode: "markers",
          name,
          // A corpus is mostly one- and two-print clusters; listing all of
          // them buries the voices that matter and covers the plot. The tail
          // still draws in its own colour, and the inspector still names it.
          showlegend: index < LEGEND_LIMIT,
          x: points.map((point) => point.coords[ax ?? 0]),
          y: points.map((point) => point.coords[ay ?? 1]),
          ...(is3d ? { z: points.map((point) => point.coords[az ?? 2]) } : {}),
          customdata: points.map((point) => [
            point.artifact_id,
            point.session_label ?? shortId(point.session_id),
            point.talk_ms ? (point.talk_ms / 1000).toFixed(1) : "?",
            point.distance === null || point.distance === undefined
              ? "—"
              : point.distance.toFixed(3),
          ]),
          marker: {
            size: points.map((point) => markerSize(point.talk_ms, is3d)),
            color: unassigned ? "rgba(141,150,168,.45)" : `hsl(${hue(key)} 55% 55%)`,
            ...(is3d
              ? { opacity: unassigned ? 0.45 : 0.9 }
              : {
                  line: {
                    width: unassigned ? 1 : 0.5,
                    color: unassigned ? MUTED : "rgba(0,0,0,.45)",
                  },
                }),
          },
          hovertemplate:
            `<b>${name}</b><br>%{customdata[1]}<br>` +
            "speech %{customdata[2]}s · distance %{customdata[3]}<extra></extra>",
        }
      })
  }, [data, view, is3d])

  const layout = useMemo(() => {
    const axis = (index: number) => ({
      title: { text: `PC${index + 1}` },
      gridcolor: GRID,
      zerolinecolor: "#272d38",
      ...(is3d ? { backgroundcolor: "#0b0d12", showbackground: true } : {}),
    })
    const [ax, ay, az] = view.axes

    return {
      // Plotly keeps the user's zoom/pan (and 3D camera) while uirevision is
      // unchanged. The basis id moves exactly when the axes stop meaning the
      // same thing; the view key is in it so switching pairs resets the frame.
      uirevision: `${data?.basis_id ?? "empty"}:${view.key}`,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "#0b0d12",
      font: { color: MUTED, size: 11 },
      margin: is3d
        ? { l: 0, r: 176, t: 0, b: 0 }
        : { l: 48, r: 176, t: 12, b: 40 },
      hovermode: "closest",
      showlegend: true,
      // Outside the plotting area: the points must not hide behind a legend.
      legend: {
        x: 1.015,
        xanchor: "left",
        y: 1,
        yanchor: "top",
        bgcolor: "rgba(0,0,0,0)",
        title: { text: `largest voices (${LEGEND_LIMIT})`, font: { size: 10 } },
      },
      ...(is3d
        ? {
            scene: {
              xaxis: axis(ax ?? 0),
              yaxis: axis(ay ?? 1),
              zaxis: axis(az ?? 2),
              // "data", not "cube": the components share units but not
              // ranges, so a forced cube would stretch PC3 and overstate how
              // far apart voices are along it.
              aspectmode: "data",
              camera: { eye: { x: 1.05, y: 1.05, z: 0.85 } },
            },
          }
        : {
            dragmode: "pan",
            xaxis: axis(ax ?? 0),
            yaxis: { ...axis(ay ?? 1), scaleanchor: "x" },
          }),
    }
  }, [data?.basis_id, view, is3d])

  if (error) return <div className="banner">{error}</div>
  if (!data || !Plot) return <div className="loading">loading the voice map…</div>
  if (!data.model) return <div className="empty">no voice-prints yet</div>

  const parts = view.axes.map((index) => ({
    label: `PC${index + 1}`,
    share: data.explained_variance_ratio[index] ?? 0,
  }))
  const total = parts.reduce((sum, part) => sum + part.share, 0)

  return (
    <div className="map-view">
      <div className="map-toolbar">
        <div className="mode-toggle">
          {VIEWS.map((option) => (
            <button
              key={option.key}
              className={`mode ${option.key === view.key ? "active" : ""}`}
              onClick={() => setView(option)}
            >
              {option.label}
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
        {parts.map((part, index) => (
          <span key={part.label}>
            {index > 0 && " + "}
            {part.label}{" "}
            <span className="row-dim">{(part.share * 100).toFixed(1)}%</span>
          </span>
        ))}{" "}
        = <strong>{(total * 100).toFixed(1)}%</strong> of the variance across{" "}
        {data.fit_points} voice-prints. The other{" "}
        {((1 - total) * 100).toFixed(1)}% is not on screen — voices that look
        close here may not be. Click a point for its true distance.
      </p>

      <div className="map-body">
        <div className={`map-plot ${is3d ? "is-3d" : ""}`}>
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
