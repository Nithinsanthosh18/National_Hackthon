import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
    https: {
      key: path.resolve(__dirname, "key.pem"),
      cert: path.resolve(__dirname, "cert.pem"),
    },
    hmr: {
      overlay: false,
    },
    proxy: {
      "/api": {
        target: "https://127.0.0.1:8001",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/video-feed": {
        target: "https://127.0.0.1:8001",
        changeOrigin: true,
        secure: false,
      },
      "/heatmap-feed": {
        target: "https://127.0.0.1:8001",
        changeOrigin: true,
        secure: false,
      },
      "/alerts": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
      "/events": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
      "/cameras": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
      "/watchlist": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
      "/stats": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
      "/archive": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
      "/uploads": { target: "https://127.0.0.1:8001", changeOrigin: true, secure: false },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
