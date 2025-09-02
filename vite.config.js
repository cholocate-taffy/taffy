import { defineConfig } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
    base: '/taffy/',
    build: {
        chunkSizeWarningLimit: 1600, 
    }
});