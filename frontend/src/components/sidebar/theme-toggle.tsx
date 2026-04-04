"use client";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
  };

  return (
    <button
      onClick={toggle}
      className="flex items-center gap-1.5 font-sans text-[13px] text-text-muted hover:text-accent transition-colors py-1"
    >
      <span className="text-sm">{isDark ? "\u2600" : "\u263E"}</span>
      {isDark ? "Light mode" : "Dark mode"}
    </button>
  );
}
