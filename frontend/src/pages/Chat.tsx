import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { FormEvent, KeyboardEvent } from "react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api, errorMessage } from "../api/client";
import type { Citation, Conversation, Message, Page, SearchHit } from "../api/types";
import { AnswerText, CitationPanel, FeedbackBar } from "../components/Answer";
import { Button, ErrorState, SkeletonRows } from "../components/ui";
import { streamMessage } from "../lib/sse";

const CONVERSATIONS = ["conversations"];

interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  confidence: Message["confidence"];
  followups: string[];
  /** False while streaming. Feedback needs a stored message. */
  complete: boolean;
}

function toTurn(message: Message): Turn {
  return {
    id: message.id,
    role: message.role === "user" ? "user" : "assistant",
    content: message.content,
    citations: message.citations,
    confidence: message.confidence,
    followups: [],
    complete: true,
  };
}

function ConversationRow({ conversation }: { conversation: Conversation }) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { conversationId } = useParams();
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(conversation.title ?? "");

  const rename = useMutation({
    mutationFn: () => api.patch(`/conversations/${conversation.id}`, { title }),
    onMutate: () => {
      setEditing(false);
      queryClient.setQueryData<Page<Conversation>>(CONVERSATIONS, (page) =>
        page && {
          ...page,
          items: page.items.map((item) =>
            item.id === conversation.id ? { ...item, title } : item,
          ),
        },
      );
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: CONVERSATIONS }),
  });

  const remove = useMutation({
    mutationFn: () => api.delete(`/conversations/${conversation.id}`),
    onMutate: () => {
      queryClient.setQueryData<Page<Conversation>>(CONVERSATIONS, (page) =>
        page && {
          total: page.total - 1,
          items: page.items.filter((item) => item.id !== conversation.id),
        },
      );
      if (conversationId === conversation.id) navigate("/chat");
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: CONVERSATIONS }),
  });

  const label = conversation.title ?? "New conversation";
  if (editing) {
    return (
      <form
        className="flex gap-1 p-1"
        onSubmit={(event) => {
          event.preventDefault();
          if (title.trim()) rename.mutate();
        }}
      >
        <input
          aria-label="Conversation title"
          autoFocus
          className="min-h-10 w-full field px-2 text-sm"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
        <Button type="submit">Save</Button>
      </form>
    );
  }
  return (
    <li className="flex items-center gap-1">
      <Link
        to={`/chat/${conversation.id}`}
        aria-current={conversationId === conversation.id ? "page" : undefined}
        className={`min-w-0 flex-1 truncate rounded px-3 py-2.5 text-sm transition-colors ${
          conversationId === conversation.id ? "bg-accent-soft font-medium text-accent-soft-ink" : "hover:bg-paper-3"
        }`}
      >
        {label}
      </Link>
      <Button variant="ghost" aria-label={`Rename ${label}`} onClick={() => setEditing(true)}>
        Edit
      </Button>
      <Button
        variant="ghost"
        aria-label={`Delete ${label}`}
        onClick={() => {
          if (window.confirm(`Delete "${label}"?`)) remove.mutate();
        }}
      >
        Delete
      </Button>
    </li>
  );
}

function Sidebar() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();
  const conversations = useQuery({
    queryKey: CONVERSATIONS,
    queryFn: () =>
      api.get<Page<Conversation>>("/conversations?limit=50").then((response) => response.data),
  });
  const search = useQuery({
    queryKey: ["message-search", query],
    enabled: query.trim().length > 1,
    queryFn: () =>
      api
        .get<Page<SearchHit>>("/conversations/search", { params: { q: query } })
        .then((response) => response.data),
  });

  return (
    <aside aria-label="Conversations" className="w-full md:w-72 md:shrink-0 md:border-r md:border-rule md:pr-6">
      <Button variant="primary" className="mb-3 w-full" onClick={() => navigate("/chat")}>
        New conversation
      </Button>
      <input
        type="search"
        aria-label="Search my messages"
        placeholder="Search my messages"
        className="mb-3 min-h-10 w-full field px-3 text-sm"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      {query.trim().length > 1 ? (
        <ul aria-label="Search results" className="space-y-1">
          {search.isLoading && <SkeletonRows rows={2} label="Searching" />}
          {search.data?.items.length === 0 && (
            <li className="p-2 text-sm text-muted">No messages match.</li>
          )}
          {search.data?.items.map((hit) => (
            <li key={hit.message_id}>
              <Link
                to={`/chat/${hit.conversation_id}`}
                className="block rounded p-2 text-sm hover:bg-paper-3"
              >
                {hit.snippet.replace(/<\/?mark>/g, "")}
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <>
          {conversations.isLoading && <SkeletonRows label="Loading conversations" />}
          {conversations.isError && (
            <ErrorState error={conversations.error} onRetry={() => void conversations.refetch()} />
          )}
          {conversations.data?.items.length === 0 && (
            <p className="p-2 text-sm text-muted">Your conversations will appear here.</p>
          )}
          <ul className="max-h-[60dvh] space-y-1 overflow-y-auto">
            {conversations.data?.items.map((conversation) => (
              <ConversationRow key={conversation.id} conversation={conversation} />
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const stopper = useRef<AbortController | null>(null);
  const skipLoadFor = useRef<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const history = useQuery({
    queryKey: ["messages", conversationId],
    // A brand new conversation is already on screen, and is being written by the stream.
    enabled: Boolean(conversationId) && skipLoadFor.current !== conversationId,
    queryFn: () =>
      api
        .get<Page<Message>>(`/conversations/${conversationId}/messages`)
        .then((response) => response.data),
  });

  useEffect(() => {
    if (!conversationId) setTurns([]);
    else if (history.data) setTurns(history.data.items.map(toTurn));
  }, [conversationId, history.data]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [turns]);

  const updateLast = (change: (turn: Turn) => Turn) =>
    setTurns((current) => [...current.slice(0, -1), change(current[current.length - 1])]);

  const send = async (content: string) => {
    if (!content.trim() || streaming) return;
    setDraft("");
    setProblem(null);
    setStreaming(true);
    const controller = new AbortController();
    stopper.current = controller;
    setTurns((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content, citations: [], confidence: null, followups: [], complete: true },
      { id: "streaming", role: "assistant", content: "", citations: [], confidence: null, followups: [], complete: false },
    ]);

    try {
      let id = conversationId;
      if (!id) {
        const created = await api.post<Conversation>("/conversations", {});
        id = created.data.id;
        skipLoadFor.current = id;
        navigate(`/chat/${id}`, { replace: true });
      }
      await streamMessage(
        id,
        content,
        (event) => {
          if (event.event === "meta") {
            updateLast((turn) => ({ ...turn, id: event.data.message_id }));
          } else if (event.event === "token") {
            updateLast((turn) => ({ ...turn, content: turn.content + event.data.text }));
          } else if (event.event === "citations") {
            updateLast((turn) => ({
              ...turn,
              citations: event.data.citations,
              confidence: event.data.confidence,
              followups: event.data.suggested_followups,
            }));
          } else if (event.event === "done") {
            updateLast((turn) => ({ ...turn, complete: true }));
          } else if (event.event === "error") {
            setProblem(event.data.message);
          }
        },
        controller.signal,
      );
    } catch (error) {
      if (!controller.signal.aborted) setProblem(errorMessage(error));
    } finally {
      setStreaming(false);
      stopper.current = null;
      void queryClient.invalidateQueries({ queryKey: CONVERSATIONS });
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void send(draft);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send(draft);
    }
  };

  return (
    <div className="flex flex-col gap-8 md:flex-row">
      <Sidebar />
      <section aria-label="Conversation" className="flex min-w-0 flex-1 flex-col">
        {history.isLoading && <SkeletonRows label="Loading messages" />}
        {history.isError && <ErrorState error={history.error} />}
        {!conversationId && turns.length === 0 && (
          <div className="py-10">
            <h1 className="page-title max-w-[18ch]">Ask about your college</h1>
            <p className="mt-4 max-w-prose text-lg text-muted">
              Answers come from official documents, with their sources. You can also ask me to set
              a reminder, plan your studies or summarize notes.
            </p>
          </div>
        )}

        <div role="log" aria-live="polite" aria-busy={streaming} className="flex-1 space-y-5">
          {turns.map((turn, index) =>
            turn.role === "user" ? (
              <p key={turn.id} className="ml-auto w-fit max-w-[85%] rounded-card rounded-br bg-paper-3 px-4 py-2.5">
                {turn.content}
              </p>
            ) : (
              <article key={turn.id} className="panel p-6">
                {turn.content ? (
                  <AnswerText
                    text={turn.content}
                    citations={turn.citations}
                    onOpenCitation={setOpenCitation}
                  />
                ) : (
                  <div className="skeleton h-6 w-1/2" role="status" aria-label="Thinking" />
                )}
                {turn.confidence === "low" && turn.content && (
                  <p className="mt-2 text-sm text-muted">
                    Low confidence. Check this against the document itself.
                  </p>
                )}
                {turn.complete && index === turns.length - 1 && turn.followups.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {turn.followups.map((followup) => (
                      <Button key={followup} onClick={() => void send(followup)}>
                        {followup}
                      </Button>
                    ))}
                  </div>
                )}
                {turn.complete && <FeedbackBar messageId={turn.id} />}
              </article>
            ),
          )}
          <div ref={endRef} />
        </div>

        {problem && (
          <p role="alert" className="mt-3 text-error">
            {problem}
          </p>
        )}

        <form onSubmit={submit} className="panel sticky bottom-4 mt-6 flex items-end gap-2 p-2">
          <textarea
            aria-label="Your message"
            rows={2}
            placeholder="Ask a question"
            className="min-h-10 flex-1 resize-y rounded border-0 bg-transparent p-2 text-ink"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
          />
          {streaming ? (
            <Button variant="danger" onClick={() => stopper.current?.abort()}>
              Stop generating
            </Button>
          ) : (
            <Button type="submit" variant="primary" disabled={!draft.trim()}>
              Send
            </Button>
          )}
        </form>
      </section>
      {openCitation && (
        <CitationPanel citation={openCitation} onClose={() => setOpenCitation(null)} />
      )}
    </div>
  );
}
