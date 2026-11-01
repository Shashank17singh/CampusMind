import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { linkCitations } from "../components/Answer";
import * as sse from "../lib/sse";
import { parseEvents, type ChatEvent } from "../lib/sse";
import { ChatPage } from "../pages/Chat";

const CITATION = {
  number: 2,
  document_id: "d1",
  title: "Fee Circular",
  page: 4,
  snippet: "The fee is 800 rupees.",
};

function renderChat(path = "/chat") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/chat/:conversationId?" element={<ChatPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

function mockApi() {
  vi.spyOn(api, "get").mockImplementation((url: string) => {
    if (url.includes("/messages")) {
      return Promise.resolve({ data: { items: [], total: 0 } });
    }
    return Promise.resolve({
      data: {
        items: [
          { id: "c1", title: "Exam fees", created_at: "", last_message_at: "" },
        ],
        total: 1,
      },
    });
  });
  vi.spyOn(api, "post").mockResolvedValue({ data: { id: "c2" } });
}

function streamEvents(events: ChatEvent[]) {
  vi.spyOn(sse, "streamMessage").mockImplementation(async (_id, _content, onEvent) => {
    events.forEach(onEvent);
  });
}

const ANSWER: ChatEvent[] = [
  { event: "meta", data: { trace_id: "t", message_id: "m1" } },
  { event: "token", data: { text: "The fee " } },
  { event: "token", data: { text: "is 800 [2]." } },
  {
    event: "citations",
    data: { citations: [CITATION], confidence: "high", suggested_followups: ["When is it due?"] },
  },
  { event: "done", data: { intent: "qa", latency_ms: 10 } },
];

describe("chat", () => {
  beforeEach(mockApi);

  it("lists conversations and shows the empty state", async () => {
    renderChat();
    expect(await screen.findByRole("link", { name: "Exam fees" })).toBeInTheDocument();
    expect(screen.getByText("Ask about your college")).toBeInTheDocument();
  });

  it("streams an answer into the conversation", async () => {
    streamEvents(ANSWER);
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "What is the fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    const log = screen.getByRole("log");
    expect(await within(log).findByText("What is the fee?")).toBeInTheDocument();
    expect(await within(log).findByText(/The fee is 800/)).toBeInTheDocument();
  });

  it("opens the right source from a citation chip", async () => {
    streamEvents(ANSWER);
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(await screen.findByRole("button", { name: "Source 2: Fee Circular" }));
    const panel = screen.getByRole("complementary", { name: "Source" });
    expect(within(panel).getByText("The fee is 800 rupees.")).toBeInTheDocument();
    expect(within(panel).getByText("Page 4")).toBeInTheDocument();
    await user.click(within(panel).getByRole("button", { name: "Close source" }));
    expect(screen.queryByRole("complementary", { name: "Source" })).not.toBeInTheDocument();
  });

  it("sends a suggested follow-up when pressed", async () => {
    streamEvents(ANSWER);
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(await screen.findByRole("button", { name: "When is it due?" }));
    await waitFor(() => expect(sse.streamMessage).toHaveBeenCalledTimes(2));
    expect(vi.mocked(sse.streamMessage).mock.calls[1][1]).toBe("When is it due?");
  });

  it("offers reasons on a thumbs down and posts the choice", async () => {
    streamEvents(ANSWER);
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(await screen.findByRole("button", { name: "Not helpful" }));
    await user.click(screen.getByRole("button", { name: "Outdated" }));
    expect(api.post).toHaveBeenCalledWith("/messages/m1/feedback", {
      rating: -1,
      reason: "outdated",
    });
  });

  it("puts a failed rating back and says so", async () => {
    streamEvents(ANSWER);
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    const helpful = await screen.findByRole("button", { name: "Helpful" });
    vi.mocked(api.post).mockRejectedValueOnce(new Error("down"));
    await user.click(helpful);
    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong");
    expect(helpful).toHaveAttribute("aria-pressed", "false");
  });

  it("aborts the stream when stop is pressed", async () => {
    let signal: AbortSignal | undefined;
    vi.spyOn(sse, "streamMessage").mockImplementation(
      (_id, _content, onEvent, aborter) =>
        new Promise((_resolve, reject) => {
          signal = aborter;
          onEvent({ event: "token", data: { text: "Partial" } });
          aborter.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    );
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await user.click(await screen.findByRole("button", { name: "Stop generating" }));
    expect(signal?.aborted).toBe(true);
    expect(await screen.findByRole("button", { name: "Send" })).toBeInTheDocument();
    expect(screen.getByText("Partial")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the server's message when streaming reports an error", async () => {
    streamEvents([
      {
        event: "error",
        data: { code: "llm_unavailable", message: "The assistant is unavailable.", trace_id: "t" },
      },
    ]);
    const user = userEvent.setup();
    renderChat("/chat/c1");
    await user.type(screen.getByLabelText("Your message"), "fee?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The assistant is unavailable.");
  });

  it("creates a conversation before the first message", async () => {
    streamEvents(ANSWER);
    const user = userEvent.setup();
    renderChat();
    await user.type(screen.getByLabelText("Your message"), "fee?{Enter}");
    await waitFor(() => expect(sse.streamMessage).toHaveBeenCalled());
    expect(api.post).toHaveBeenCalledWith("/conversations", {});
    expect(vi.mocked(sse.streamMessage).mock.calls[0][0]).toBe("c2");
  });
});

describe("streaming parser", () => {
  it("keeps an incomplete event for the next read", () => {
    const first = parseEvents('event: token\ndata: {"text":"a"}\n\nevent: tok');
    expect(first.events).toEqual([{ event: "token", data: { text: "a" } }]);
    expect(first.rest).toBe("event: tok");
    const second = parseEvents(`${first.rest}en\ndata: {"text":"b"}\n\n`);
    expect(second.events).toEqual([{ event: "token", data: { text: "b" } }]);
  });
});

describe("citation links", () => {
  it("links bare markers only", () => {
    expect(linkCitations("See [1] and [text](http://x) and [2](#a)")).toBe(
      "See [1](#cite-1) and [text](http://x) and [2](#a)",
    );
  });
});
