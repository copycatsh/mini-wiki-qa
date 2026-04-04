import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        serif: ["'Instrument Serif'", "Georgia", "serif"],
        body: ["'Source Sans 3'", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "Consolas", "monospace"],
      },
      colors: {
        background: "var(--background)",
        surface: "var(--surface)",
        sidebar: "var(--sidebar)",
        "code-bg": "var(--code-bg)",
        border: "var(--border)",
        text: {
          DEFAULT: "var(--text)",
          muted: "var(--text-muted)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          hover: "var(--accent-hover)",
        },
        success: "var(--success)",
        warning: "var(--warning)",
        error: "var(--error)",
        info: "var(--info)",
      },
      maxWidth: {
        chat: "720px",
      },
      width: {
        sidebar: "240px",
      },
      borderRadius: {
        sm: "4px",
        md: "8px",
        lg: "12px",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
