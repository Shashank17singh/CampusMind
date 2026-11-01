import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import { API_BASE } from "../api/client";
import { useAuth } from "../stores/auth";
import { useUi } from "../stores/ui";

const RECONNECT_MS = 5000;
const CLOSE_UNAUTHORIZED = 4401;

/** Hold the notifications socket open while signed in, and show arrivals as toasts. */
export function useNotifications(): void {
  const token = useAuth((state) => state.accessToken);
  const pushToast = useUi((state) => state.pushToast);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!token) return;
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;

    const connect = () => {
      const scheme = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${window.location.host}${API_BASE}/ws/notifications`);
      socket.onopen = () => socket?.send(JSON.stringify({ type: "auth", token }));
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data as string) as {
          type: string;
          title?: string;
          body?: string;
        };
        if (message.type === "notification") {
          pushToast(message.title ?? "Reminder", message.body);
          void queryClient.invalidateQueries({ queryKey: ["reminders"] });
        }
      };
      socket.onclose = (event) => {
        // A new token re-runs this effect, so an auth close does not retry by itself.
        if (!stopped && event.code !== CLOSE_UNAUTHORIZED) {
          retry = setTimeout(connect, RECONNECT_MS);
        }
      };
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, [token, pushToast, queryClient]);
}
