import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

// Installable PWA: on iPhone "Teilen → Zum Home-Bildschirm" gives a full-screen
// app that works offline (the strategy nets are cached on first use).
export default defineConfig({
  base: "./",
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icon.svg", "apple-touch-icon.png"],
      manifest: {
        name: "SkullKI – Skull King GTO",
        short_name: "SkullKI",
        description: "GTO-Solver für Skull King (4 Spieler)",
        lang: "de",
        theme_color: "#071a1f",
        background_color: "#071a1f",
        display: "standalone",
        orientation: "portrait",
        icons: [
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,wasm,svg,png,woff2}"],
        maximumFileSizeToCacheInBytes: 8 * 1024 * 1024,
        runtimeCaching: [
          {
            urlPattern: /\/models\/r\d+\.bin$/,
            handler: "CacheFirst",
            options: { cacheName: "skullki-models", expiration: { maxEntries: 20 } },
          },
        ],
      },
    }),
  ],
});
