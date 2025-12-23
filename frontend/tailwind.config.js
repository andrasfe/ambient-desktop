/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Deep space theme
        void: {
          50: '#f0f4ff',
          100: '#e0e8ff',
          200: '#c7d4ff',
          300: '#a3b6ff',
          400: '#7a8eff',
          500: '#5c66ff',
          600: '#4f46f5',
          700: '#4338d8',
          800: '#372fae',
          900: '#312c89',
          950: '#0a0a1a',
        },
        neon: {
          cyan: '#00ffff',
          pink: '#ff00ff',
          green: '#00ff88',
          orange: '#ff8800',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Space Grotesk', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 255, 255, 0.3)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 255, 255, 0.6)' },
        },
      },
    },
  },
  plugins: [],
}

