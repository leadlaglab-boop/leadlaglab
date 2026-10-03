import { defineConfig } from "astro/config";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  site: "https://leadlaglab.com",
  output: "static",
  compressHTML: true,
  build: {
    assets: "_assets",
  },
  vite: {
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src"),
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes("@observablehq/plot")) return "plot";
          },
        },
      },
    },
  },
});
