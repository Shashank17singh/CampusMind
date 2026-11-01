import { create } from "zustand";

type ThemeName = "light" | "dark";

interface Toast {
  id: number;
  title: string;
  body?: string;
}

interface UiState {
  theme: ThemeName;
  toasts: Toast[];
  toggleTheme: () => void;
  pushToast: (title: string, body?: string) => void;
  dismissToast: (id: number) => void;
}

const TOAST_MS = 6000;
let nextToast = 1;

function readTheme(): ThemeName {
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // Storage can be blocked. The system preference still works.
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme;
}

export const useUi = create<UiState>((set, get) => ({
  theme: readTheme(),
  toasts: [],
  toggleTheme: () => {
    const theme = get().theme === "dark" ? "light" : "dark";
    applyTheme(theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // The choice lasts this session if storage is blocked.
    }
    set({ theme });
  },
  pushToast: (title, body) => {
    const id = nextToast++;
    set((state) => ({ toasts: [...state.toasts, { id, title, body }] }));
    setTimeout(() => get().dismissToast(id), TOAST_MS);
  },
  dismissToast: (id) =>
    set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}));
