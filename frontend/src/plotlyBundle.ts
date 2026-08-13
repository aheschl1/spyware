// A custom Plotly bundle: core plus only the two traces this app draws.
// The prebuilt dists are all-or-nothing — gl2d has no scatter3d, and the full
// dist is ~3x the size for traces nothing renders.
import Plotly from "plotly.js/lib/core"
import scatter3d from "plotly.js/lib/scatter3d"
import scattergl from "plotly.js/lib/scattergl"

Plotly.register([scattergl, scatter3d])

export default Plotly
