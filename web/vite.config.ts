import fs from "node:fs";
import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "copy-public-without-generated",
      closeBundle() {
        const pubDir = path.resolve(__dirname, "public");
        const outDir = path.resolve(__dirname, "dist");
        if (!fs.existsSync(pubDir) || !fs.existsSync(outDir)) return;
        for (const entry of fs.readdirSync(pubDir, { withFileTypes: true })) {
          if (entry.name === "generated") continue;
          const src = path.join(pubDir, entry.name);
          const dest = path.join(outDir, entry.name);
          fs.cpSync(src, dest, { recursive: true });
        }
      },
    },
  ],
  build: {
    copyPublicDir: false,
  },
  server: {
    proxy: {
      "/api": apiProxyTarget,
      "/generated": apiProxyTarget,
    },
    watch: {
      // The backend writes tens of thousands of export files into public/generated (served via
      // the /generated proxy above). Crawling them pins Vite's CPU on startup and blocks requests.
      ignored: ["**/public/generated/**", "**/dist/**", "**/artifacts/**"],
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    testTimeout: 10000,
    poolOptions: {
      forks: {
        execArgv: ["--max-old-space-size=4096"],
      },
    },
  },
});
