import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // dialkit and motion each import react; without deduping, Vite can hand them a
  // separate React instance than the app, triggering "Invalid hook call".
  resolve: {
    dedupe: ['react', 'react-dom'],
  },
})
