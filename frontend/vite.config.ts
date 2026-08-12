import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev pairing with `make api`: the browser talks same-origin and Vite relays
// to the local API, so CORS and token-in-URL audio both just work. API_PROXY
// points elsewhere when the API is bound to another interface (e.g.
// `make api HOST=10.8.0.1` → API_PROXY=http://10.8.0.1:8000).
const target = process.env.API_PROXY ?? "http://127.0.0.1:8000"

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/v1": target,
      "/health": target,
    },
  },
})
