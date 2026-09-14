export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#f6f2e9",
        "bg-2": "#efe9dc",
        surface: "#fdfbf6",
        "surface-2": "#f3eee3",
        ink: "#2a2722",
        "ink-2": "#5d564c",
        "ink-3": "#8c8475",
        line: "#e2dac9",
        "line-2": "#d3c9b4",
        terracotta: "#c0764a",
        "terracotta-soft": "#e7c8af",
        sage: "#7e8b66",
        "sage-soft": "#cdd4bd",
        cobalt: "#4f7196",
        "cobalt-deep": "#3d5a7a",
        "cobalt-soft": "#c2cfdd",
        ok: "#6f8a5b",
        warn: "#c0764a",
        bad: "#b0563f",
      },
      fontFamily: {
        ui: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        editorial: ["Newsreader", "Georgia", "serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm: "7px",
        DEFAULT: "10px",
        lg: "16px",
      },
      boxShadow: {
        sm: "0 1px 2px rgba(60,50,35,.05), 0 1px 1px rgba(60,50,35,.04)",
        DEFAULT: "0 2px 8px rgba(60,50,35,.06), 0 1px 2px rgba(60,50,35,.05)",
        lg: "0 18px 50px -18px rgba(50,42,30,.30), 0 6px 16px -8px rgba(50,42,30,.18)",
      },
    },
  },
  plugins: [],
}
