import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Dev pairing with `make api`: the browser talks same-origin and Vite relays
// to the local API, so CORS and token-in-URL audio both just work.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
})
