import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发环境把 /api 代理到后端，避免跨域并让 EventSource 直接使用同源地址
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
