import { API_BASE, refreshAccessToken } from "../api/client";
import type { Citation } from "../api/types";
import { useAuth } from "../stores/auth";

export type ChatEvent =
  | { event: "meta"; data: { trace_id: string; message_id: string } }
  | { event: "token"; data: { text: string } }
  | {
      event: "citations";
      data: {
        citations: Citation[];
        confidence: "high" | "medium" | "low" | null;
        suggested_followups: string[];
      };
    }
  | { event: "done"; data: { intent: string | null; latency_ms: number } }
  | { event: "error"; data: { code: string; message: string; trace_id: string } };

/** Split a Server-Sent Events buffer, returning complete events and the unfinished rest. */
export function parseEvents(buffer: string): { events: ChatEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: ChatEvent[] = [];
  for (const block of blocks) {
    const name = block.match(/^event: (.+)$/m)?.[1];
    const data = block.match(/^data: (.+)$/m)?.[1];
    if (name && data) events.push({ event: name, data: JSON.parse(data) } as ChatEvent);
  }
  return { events, rest };
}

function post(path: string, body: unknown, signal: AbortSignal): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${useAuth.getState().accessToken ?? ""}`,
    },
    body: JSON.stringify(body),
    signal,
  });
}

/**
 * Post a message and read the event stream. `fetch` is used because EventSource can only
 * send GET and cannot set an Authorization header. Aborting the signal is the stop button.
 */
export async function streamMessage(
  conversationId: string,
  content: string,
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const path = `/conversations/${conversationId}/messages`;
  let response = await post(path, { content }, signal);
  if (response.status === 401) {
    await refreshAccessToken();
    response = await post(path, { content }, signal);
  }
  if (!response.ok || !response.body) {
    throw new Error("The message could not be sent. Please try again.");
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += value;
    const { events, rest } = parseEvents(buffer);
    buffer = rest;
    events.forEach(onEvent);
  }
}
