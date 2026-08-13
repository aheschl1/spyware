// plotly.js's own lib/* entry points ship no types; react-plotly.js's factory
// only needs the module object, so an opaque default is enough.
declare module "plotly.js/lib/core" {
  const Plotly: { register: (traces: unknown[]) => void }
  export default Plotly
}
declare module "plotly.js/lib/scattergl" {
  const trace: unknown
  export default trace
}
declare module "plotly.js/lib/scatter3d" {
  const trace: unknown
  export default trace
}
