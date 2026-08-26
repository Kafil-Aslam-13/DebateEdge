import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Dev only: proxy /api → backend
      "/api": {
        target: process.env.VITE_BACKEND_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir:    "dist",
    sourcemap: false,    // disable in prod for smaller bundle
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          axios: ["axios"],
        },
      },
    },
  },
});