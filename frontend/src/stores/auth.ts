import { create } from "zustand";

import type { User } from "../api/types";

interface AuthState {
  /** Kept in memory only, so an XSS bug cannot read it from storage. */
  accessToken: string | null;
  user: User | null;
  /** False until the first silent refresh has finished. */
  ready: boolean;
  setSession: (accessToken: string, user: User) => void;
  setToken: (accessToken: string) => void;
  setUser: (user: User) => void;
  setReady: () => void;
  clear: () => void;
}

export const useAuth = create<AuthState>((set) => ({
  accessToken: null,
  user: null,
  ready: false,
  setSession: (accessToken, user) => set({ accessToken, user }),
  setToken: (accessToken) => set({ accessToken }),
  setUser: (user) => set({ user }),
  setReady: () => set({ ready: true }),
  clear: () => set({ accessToken: null, user: null }),
}));
