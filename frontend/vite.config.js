import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server proxies /api straight to Application Service — the browser only
// ever talks to one origin (this dev server), so no CORS setup is needed
// anywhere in the FastAPI services.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
