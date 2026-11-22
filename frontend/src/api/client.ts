import axios, { AxiosError, type InternalAxiosRequestConfig } from "axios";

import { useAuth } from "../stores/auth";

export const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)
  ? `${import.meta.env.VITE_API_URL}/api/v1`
  : "/api/v1";

export const api = axios.create({ baseURL: API_BASE, withCredentials: true });

let refreshing: Promise<string> | null = null;

const refreshChannel =
  typeof window !== "undefined" && "BroadcastChannel" in window
    ? new BroadcastChannel("campusmind_token_refresh")
    : null;

if (refreshChannel) {
  refreshChannel.onmessage = (event: MessageEvent<{ type: string; token?: string }>) => {
    if (event.data?.type === "REFRESH_SUCCESS" && event.data.token) {
      useAuth.getState().setToken(event.data.token);
    }
  };
}

/**
 * Get a new access token. Refresh tokens are single use and reuse signs the user out, so
 * every caller waiting at the same moment must share one request across calls and tabs.
 */
export function refreshAccessToken(): Promise<string> {
  refreshing ??= axios
    .post<{ access_token: string }>(`${API_BASE}/auth/refresh`, null, { withCredentials: true })
    .then((response) => {
      const token = response.data.access_token;
      useAuth.getState().setToken(token);
      refreshChannel?.postMessage({ type: "REFRESH_SUCCESS", token });
      return token;
    })
    .finally(() => {
      refreshing = null;
    });
  return refreshing;
}

api.interceptors.request.use((config) => {
  const token = useAuth.getState().accessToken;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

interface RetriableConfig extends InternalAxiosRequestConfig {
  retried?: boolean;
}

api.interceptors.response.use(undefined, async (error: AxiosError) => {
  const original = error.config as RetriableConfig | undefined;
  const isAuthCall = original?.url?.startsWith("/auth/");
  if (error.response?.status !== 401 || !original || original.retried || isAuthCall) {
    throw error;
  }
  original.retried = true;
  try {
    const token = await refreshAccessToken();
    original.headers.Authorization = `Bearer ${token}`;
    return await api.request(original);
  } catch {
    useAuth.getState().clear();
    throw error;
  }
});

/** Read the message from the API's standard error shape. */
export function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const body = error.response?.data as { error?: { message?: string } } | undefined;
    if (body?.error?.message) return body.error.message;
    if (!error.response) return "Can't reach the server. Check your connection and try again.";
  }
  return "Something went wrong. Please try again.";
}
