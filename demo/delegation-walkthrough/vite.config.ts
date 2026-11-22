import path from "path"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"
import { inspectAttr } from 'kimi-plugin-inspect-react'

// https://vite.dev/config/
export default defineConfig({
  base: './',
  // 双栈绑定：vite 7 默认只监听 IPv6 ::1，本机若把 localhost 解析到
  // 127.0.0.1（IPv4）浏览器与预览卡片都连不上；显式 true 同时绑两套栈。
  server: {
    host: true,
  },
  plugins: [inspectAttr(), react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
