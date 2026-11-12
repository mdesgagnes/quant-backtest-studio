import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the API runs on :8000 (uvicorn server.app:app) and Vite proxies to it.
// Build: `npm run build` writes web/dist, which the API serves itself.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 700 },
});
