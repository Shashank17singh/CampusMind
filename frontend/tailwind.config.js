/** Every colour and font resolves to a token in src/styles/tokens.css. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    colors: {
      transparent: "transparent",
      current: "currentColor",
      paper: "var(--color-paper)",
      "paper-2": "var(--color-paper-2)",
      "paper-3": "var(--color-paper-3)",
      sheet: "var(--color-sheet)",
      ink: "var(--color-ink)",
      "ink-2": "var(--color-ink-2)",
      muted: "var(--color-muted)",
      rule: "var(--color-rule)",
      "rule-2": "var(--color-rule-2)",
      accent: "var(--color-accent)",
      "accent-2": "var(--color-accent-2)",
      "accent-ink": "var(--color-accent-ink)",
      "accent-soft": "var(--color-accent-soft)",
      "accent-soft-ink": "var(--color-accent-soft-ink)",
      error: "var(--color-error)",
      success: "var(--color-success)",
    },
    fontFamily: {
      sans: "var(--font-body)",
      display: "var(--font-display)",
      serif: "var(--font-body)",
      mono: "var(--font-outlier)",
    },
    borderRadius: {
      none: "0",
      DEFAULT: "var(--radius-part)",
      card: "var(--radius-card)",
      full: "9999px",
    },
    screens: { sm: "640px", md: "960px", lg: "1200px" },
    extend: {
      maxWidth: { content: "1200px", prose: "68ch" },
      zIndex: { sticky: "200", modal: "400", toast: "500" },
    },
  },
};
