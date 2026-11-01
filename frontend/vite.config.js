import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/keiba/",
  plugins: [react()],
  build: {
    outDir: "../pipeline/public_frontend_dist",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/keiba/api": "http://127.0.0.1:8000", "/.well-known": "http://127.0.0.1:8000" },
  },
});
