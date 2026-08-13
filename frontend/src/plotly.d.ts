// The gl2d bundle ships no types; react-plotly.js's factory only needs the
// module object, so an opaque default is enough.
declare module "plotly.js-gl2d-dist-min" {
  const Plotly: unknown
  export default Plotly
}
