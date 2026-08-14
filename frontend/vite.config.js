import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite 配置：开发服务器运行在 5173，并把 /api 请求代理到后端 FastAPI 服务
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});
