import { useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { api, errorMessage } from "../api/client";
import type { Citation, FeedbackReason } from "../api/types";
import { Button } from "./ui";

const CITE_PREFIX = "#cite-";

/** Turn [1] markers into links, which the renderer below shows as buttons. */
export function linkCitations(text: string): string {
  return text.replace(/\[(\d+)\](?!\()/g, `[$1](${CITE_PREFIX}$1)`);
}

interface AnswerTextProps {
  text: string;
  citations: Citation[];
  onOpenCitation: (citation: Citation) => void;
}

export function AnswerText({ text, citations, onOpenCitation }: AnswerTextProps) {
  return (
    <div className="prose-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a({ href, children }) {
            if (href?.startsWith(CITE_PREFIX)) {
              const number = Number(href.slice(CITE_PREFIX.length));
              const citation = citations.find((item) => item.number === number);
              if (!citation) return <sup className="text-muted">[{number}]</sup>;
              return (
                <button
                  type="button"
                  onClick={() => onOpenCitation(citation)}
                  aria-label={`Source ${number}: ${citation.title}`}
                  className="mx-0.5 inline-flex min-h-6 min-w-6 items-center justify-center rounded-full bg-accent-soft px-1.5 align-baseline text-xs font-medium tabular-nums text-accent-soft-ink transition-colors hover:bg-accent hover:text-accent-ink"
                >
                  {number}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="underline">
                {children}
              </a>
            );
          },
        }}
      >
        {linkCitations(text)}
      </ReactMarkdown>
    </div>
  );
}

export function CitationPanel({
  citation,
  onClose,
}: {
  citation: Citation;
  onClose: () => void;
}) {
  return (
    <aside
      aria-label="Source"
      className="panel fixed inset-y-0 right-0 z-modal w-full max-w-md overflow-y-auto rounded-none p-6 md:sticky md:top-24 md:z-auto md:h-fit md:max-h-[calc(100dvh-7rem)] md:w-80 md:shrink-0 md:rounded-card"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-display text-3xl tabular-nums text-accent">{citation.number}</p>
          <h2 className="mt-1 text-lg">{citation.title}</h2>
        </div>
        <Button variant="ghost" aria-label="Close source" onClick={onClose} autoFocus>
          Close
        </Button>
      </div>
      {citation.page !== null && <p className="mt-1 text-sm text-muted">Page {citation.page}</p>}
      <blockquote className="mt-5 border-l-2 border-accent pl-4 leading-relaxed text-ink-2">
        {citation.snippet}
      </blockquote>
    </aside>
  );
}

const REASONS: { value: FeedbackReason; label: string }[] = [
  { value: "wrong", label: "Wrong" },
  { value: "outdated", label: "Outdated" },
  { value: "incomplete", label: "Incomplete" },
  { value: "not_relevant", label: "Not relevant" },
];

export function FeedbackBar({ messageId }: { messageId: string }) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [asking, setAsking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const send = async (value: 1 | -1, reason?: FeedbackReason) => {
    const previous = rating;
    setRating(value);
    setAsking(false);
    setProblem(null);
    try {
      await api.post(`/messages/${messageId}/feedback`, { rating: value, reason });
    } catch (error) {
      setRating(previous);
      setProblem(errorMessage(error));
    }
  };

  return (
    <div className="relative mt-4 flex flex-wrap items-center gap-1 border-t border-rule pt-3">
      <Button
        variant="ghost"
        aria-label="Helpful"
        aria-pressed={rating === 1}
        className={rating === 1 ? "bg-paper-3" : ""}
        onClick={() => void send(1)}
      >
        Helpful
      </Button>
      <Button
        variant="ghost"
        aria-label="Not helpful"
        aria-pressed={rating === -1}
        aria-expanded={asking}
        className={rating === -1 ? "bg-paper-3" : ""}
        onClick={() => setAsking((open) => !open)}
      >
        Not helpful
      </Button>
      {asking && (
        <div role="group" aria-label="What was wrong?" className="flex flex-wrap gap-1">
          {REASONS.map((reason) => (
            <Button key={reason.value} onClick={() => void send(-1, reason.value)}>
              {reason.label}
            </Button>
          ))}
        </div>
      )}
      {problem && (
        <p role="alert" className="text-sm text-error">
          {problem}
        </p>
      )}
    </div>
  );
}
