import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api, errorMessage } from "../api/client";
import type {
  DocumentRecord,
  Page,
  Summary,
  SummaryItem,
  SummaryMode,
} from "../api/types";
import { AnswerText } from "../components/Answer";
import { Button, EmptyState, ErrorState, PageHeader, SkeletonRows } from "../components/ui";

const MODES: { value: SummaryMode; label: string }[] = [
  { value: "brief", label: "Brief, 5 points" },
  { value: "detailed", label: "Detailed, by section" },
  { value: "exam_notes", label: "Exam notes" },
  { value: "flashcards", label: "Flashcards" },
];

function pages(item: SummaryItem): string {
  return item.pages.length ? ` (p. ${item.pages.join(", ")})` : "";
}

function Flashcard({ item }: { item: SummaryItem }) {
  const [flipped, setFlipped] = useState(false);
  return (
    <button
      type="button"
      aria-pressed={flipped}
      onClick={() => setFlipped((value) => !value)}
      className="flip-card min-h-32 w-full border border-rule bg-sheet p-4 text-left hover:bg-paper-2"
    >
      <span className="block text-sm text-muted">{flipped ? "Answer" : "Question"}</span>
      <span className="mt-1 block">
        {flipped ? `${item.answer ?? ""}${pages(item)}` : item.text}
      </span>
    </button>
  );
}

function SummaryView({ summary }: { summary: Summary }) {
  const download = async (format: "md" | "pdf") => {
    const response = await api.get<Blob>(`/summaries/${summary.id}/export`, {
      params: { format },
      responseType: "blob",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(response.data);
    link.download = `summary.${format}`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const groups = summary.content?.groups ?? [];
  return (
    <section aria-label="Summary" className="mt-6">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">{summary.source_name}</h2>
        <div className="flex gap-2">
          <Button onClick={() => void download("md")}>Markdown</Button>
          <Button onClick={() => void download("pdf")}>PDF</Button>
        </div>
      </div>
      {summary.content?.verified === false && (
        <p className="mb-3 border border-error p-3 text-sm">
          This summary could not be fully checked against the source, and may be incomplete.
        </p>
      )}
      {summary.dropped_count > 0 && (
        <p className="mb-3 text-sm text-muted">
          {summary.dropped_count} items were removed because the source did not support them.
        </p>
      )}
      {groups.map((group) => (
        <div key={group.heading} className="mb-5">
          {group.heading && <h3 className="mb-2 font-semibold">{group.heading}</h3>}
          {summary.mode === "flashcards" ? (
            <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
              {group.items.map((item) => (
                <Flashcard key={item.text} item={item} />
              ))}
            </div>
          ) : (
            <ul className="space-y-2">
              {group.items.map((item) => (
                <li key={item.text}>
                  <AnswerText
                    text={`${item.text}${pages(item)}`}
                    citations={[]}
                    onOpenCitation={() => undefined}
                  />
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </section>
  );
}

export function SummarizerPage() {
  const [documentId, setDocumentId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<SummaryMode>("brief");
  const [summaryId, setSummaryId] = useState<string | null>(null);

  const documents = useQuery({
    queryKey: ["documents", "ready"],
    queryFn: () =>
      api
        .get<Page<DocumentRecord>>("/documents", { params: { status: "ready", limit: 100 } })
        .then((response) => response.data),
  });

  const start = useMutation({
    mutationFn: async () => {
      if (file) {
        const form = new FormData();
        form.append("file", file);
        form.append("mode", mode);
        const response = await api.post<{ summary_id: string }>("/summaries/upload", form);
        return response.data.summary_id;
      }
      const response = await api.post<{ summary_id: string }>("/summaries", {
        document_id: documentId,
        mode,
      });
      return response.data.summary_id;
    },
    onSuccess: setSummaryId,
  });

  const summary = useQuery({
    queryKey: ["summary", summaryId],
    enabled: Boolean(summaryId),
    queryFn: () => api.get<Summary>(`/summaries/${summaryId}`).then((response) => response.data),
    refetchInterval: (query) => (query.state.data?.status === "pending" ? 3000 : false),
  });

  return (
    <>
      <PageHeader title="Summarizer" />
      <form
        className="grid gap-4 border border-rule bg-sheet p-4 sm:grid-cols-2"
        onSubmit={(event) => {
          event.preventDefault();
          start.mutate();
        }}
      >
        <div>
          <label htmlFor="document" className="mb-1 block text-sm font-medium">
            Document
          </label>
          <select
            id="document"
            className="min-h-10 w-full rounded border border-rule-2 bg-sheet px-2"
            value={documentId}
            disabled={Boolean(file)}
            onChange={(event) => setDocumentId(event.target.value)}
          >
            <option value="">Choose a document</option>
            {documents.data?.items.map((document) => (
              <option key={document.id} value={document.id}>
                {document.title}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="own-file" className="mb-1 block text-sm font-medium">
            Or your own file
          </label>
          <input
            id="own-file"
            type="file"
            accept=".pdf,.docx,.pptx,.txt,.png,.jpg,.jpeg"
            className="min-h-10 w-full text-sm"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <p className="mt-1 text-sm text-muted">Deleted after reading. Never shared.</p>
        </div>
        <fieldset className="sm:col-span-2">
          <legend className="mb-1 text-sm font-medium">Mode</legend>
          <div className="flex flex-wrap gap-3">
            {MODES.map((option) => (
              <label key={option.value} className="flex min-h-10 items-center gap-2">
                <input
                  type="radio"
                  name="mode"
                  checked={mode === option.value}
                  onChange={() => setMode(option.value)}
                />
                {option.label}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="sm:col-span-2">
          <Button
            type="submit"
            variant="primary"
            disabled={start.isPending || (!file && !documentId)}
          >
            Summarize
          </Button>
          {start.isError && (
            <p role="alert" className="mt-2 text-error">
              {errorMessage(start.error)}
            </p>
          )}
        </div>
      </form>

      {!summaryId && (
        <div className="mt-6">
          <EmptyState title="Pick a document to begin">
            Long documents take a few minutes. You can leave this page open while it runs.
          </EmptyState>
        </div>
      )}
      {summary.isError && <ErrorState error={summary.error} />}
      {summary.data?.status === "pending" && (
        <div className="mt-6">
          <SkeletonRows label="Summarizing" />
        </div>
      )}
      {summary.data?.status === "failed" && (
        <p role="alert" className="mt-6 text-error">
          {summary.data.error_message}
        </p>
      )}
      {summary.data?.status === "ready" && <SummaryView summary={summary.data} />}
    </>
  );
}
