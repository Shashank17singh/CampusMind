import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { DragEvent, FormEvent } from "react";
import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { api, errorMessage } from "../api/client";
import type { Assignment, DocumentRecord, Page, UploadResult } from "../api/types";
import {
  Button,
  EmptyState,
  ErrorState,
  Field,
  PageHeader,
  SkeletonRows,
  Tabs,
} from "../components/ui";
import { fileSize, formatDateTime } from "../lib/format";
import { useAuth } from "../stores/auth";

const POLL_MS = 2000;
const STATUS_LABELS: Record<DocumentRecord["status"], string> = {
  uploaded: "Queued",
  extracting: "Reading",
  chunking: "Splitting",
  embedding: "Indexing",
  ready: "Ready",
  failed: "Failed",
};

interface UploadRow {
  key: string;
  name: string;
  size: number;
  /** Upload to the server, from 0 to 100. */
  sent: number;
  documentId: string | null;
  status: DocumentRecord["status"] | "sending";
  progress: number;
  note: string | null;
}

function ProgressRow({ row }: { row: UploadRow }) {
  const value = row.status === "sending" ? row.sent : row.progress;
  const label = row.status === "sending" ? "Uploading" : STATUS_LABELS[row.status];
  return (
    <li className="border border-rule bg-sheet p-3">
      <div className="flex flex-wrap justify-between gap-2 text-sm">
        <span className="min-w-0 truncate font-medium">{row.name}</span>
        <span className="text-muted">
          {fileSize(row.size)}, {label}
        </span>
      </div>
      <div
        role="progressbar"
        aria-label={`${row.name} progress`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={value}
        className="mt-2 h-1 bg-paper-3"
      >
        <div
          className={`h-full origin-left ${row.status === "failed" ? "bg-error" : "bg-accent"}`}
          style={{ transform: `scaleX(${value / 100})` }}
        />
      </div>
      {row.note && <p className="mt-2 text-sm">{row.note}</p>}
    </li>
  );
}

export function UploadPanel() {
  const [rows, setRows] = useState<UploadRow[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const patch = (key: string, change: Partial<UploadRow>) =>
    setRows((current) => current.map((row) => (row.key === key ? { ...row, ...change } : row)));

  const upload = async (file: File) => {
    const key = crypto.randomUUID();
    setRows((current) => [
      ...current,
      {
        key,
        name: file.name,
        size: file.size,
        sent: 0,
        documentId: null,
        status: "sending",
        progress: 0,
        note: null,
      },
    ]);
    const form = new FormData();
    form.append("file", file);
    try {
      const { data } = await api.post<UploadResult>("/documents", form, {
        onUploadProgress: (event) =>
          patch(key, { sent: Math.round((event.loaded / (event.total ?? file.size)) * 100) }),
      });
      patch(key, {
        documentId: data.id,
        status: data.status,
        progress: data.progress,
        note: data.duplicate_of ? "This file is already uploaded." : data.error_message,
      });
    } catch (error) {
      patch(key, { status: "failed", progress: 100, note: errorMessage(error) });
    }
  };

  const addFiles = (files: FileList | null) => {
    Array.from(files ?? []).forEach((file) => void upload(file));
  };

  const active = rows.filter(
    (row) => row.documentId && row.status !== "ready" && row.status !== "failed",
  );
  const activeIds = active.map((row) => row.documentId).join(",");

  useEffect(() => {
    if (!activeIds) return;
    const timer = setInterval(() => {
      activeIds.split(",").forEach(async (id) => {
        const { data } = await api.get<Pick<DocumentRecord, "status" | "progress" | "error_message">>(
          `/documents/${id}/status`,
        );
        setRows((current) =>
          current.map((row) =>
            row.documentId === id
              ? { ...row, status: data.status, progress: data.progress, note: data.error_message }
              : row,
          ),
        );
        if (data.status === "ready") {
          void queryClient.invalidateQueries({ queryKey: ["documents"] });
        }
      });
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [activeIds, queryClient]);

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  };

  return (
    <section aria-label="Upload documents" className="mb-8">
      <div
        data-testid="dropzone"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed p-8 text-center ${
          dragging ? "border-accent bg-paper-2" : "border-rule-2"
        }`}
      >
        <p>Drop PDF, DOCX, PPTX, TXT, PNG or JPG files here, up to 25 MB each.</p>
        <Button className="mt-3" onClick={() => inputRef.current?.click()}>
          Choose files
        </Button>
        <input
          ref={inputRef}
          type="file"
          multiple
          aria-label="Choose files to upload"
          className="sr-only"
          accept=".pdf,.docx,.pptx,.txt,.png,.jpg,.jpeg"
          onChange={(event) => {
            addFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </div>
      {rows.length > 0 && (
        <ul className="mt-4 space-y-2">
          {rows.map((row) => (
            <ProgressRow key={row.key} row={row} />
          ))}
        </ul>
      )}
    </section>
  );
}

function MyDocuments() {
  const queryClient = useQueryClient();
  const documents = useQuery({
    queryKey: ["documents", "mine"],
    queryFn: () =>
      api
        .get<Page<DocumentRecord>>("/documents", { params: { mine: true, limit: 100 } })
        .then((response) => response.data),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/documents/${id}`),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["documents"] }),
  });
  const reingest = useMutation({
    mutationFn: (id: string) => api.post(`/documents/${id}/reingest`),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["documents"] }),
  });

  if (documents.isLoading) return <SkeletonRows label="Loading documents" />;
  if (documents.isError) {
    return <ErrorState error={documents.error} onRetry={() => void documents.refetch()} />;
  }
  if (!documents.data?.items.length) {
    return <EmptyState title="No documents yet">Upload a circular or syllabus above.</EmptyState>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">My documents</caption>
        <thead>
          <tr className="border-b border-rule-2 text-left">
            <th scope="col" className="py-2 font-medium">Title</th>
            <th scope="col" className="py-2 font-medium">Type</th>
            <th scope="col" className="py-2 font-medium">Status</th>
            <th scope="col" className="py-2 text-right font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.data.items.map((document) => (
            <tr key={document.id} className="border-b border-rule">
              <td className="py-2 pr-3">{document.title}</td>
              <td className="py-2 pr-3">{document.doc_type ?? "Untagged"}</td>
              <td className="py-2 pr-3">{STATUS_LABELS[document.status]}</td>
              <td className="py-2 text-right">
                <Button variant="ghost" aria-label={`Re-ingest ${document.title}`} onClick={() => reingest.mutate(document.id)}>
                  Re-ingest
                </Button>
                <Button
                  variant="ghost"
                  aria-label={`Delete ${document.title}`}
                  onClick={() => {
                    if (window.confirm(`Delete "${document.title}"?`)) remove.mutate(document.id);
                  }}
                >
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Assignments() {
  const user = useAuth((state) => state.user);
  const queryClient = useQueryClient();
  const [form, setForm] = useState({
    title: "",
    department: user?.department ?? "",
    semester: 1,
    due_at: "",
  });
  const assignments = useQuery({
    queryKey: ["assignments"],
    queryFn: () =>
      api.get<Page<Assignment>>("/assignments?mine=true").then((response) => response.data),
  });
  const post = useMutation({
    mutationFn: () =>
      api
        .post<Assignment>("/assignments", {
          ...form,
          due_at: new Date(form.due_at).toISOString(),
        })
        .then((response) => response.data),
    onSuccess: () => {
      setForm((current) => ({ ...current, title: "", due_at: "" }));
      void queryClient.invalidateQueries({ queryKey: ["assignments"] });
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/assignments/${id}`),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["assignments"] }),
  });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    post.mutate();
  };

  return (
    <>
      <form onSubmit={submit} className="mb-6 grid gap-3 border border-rule bg-sheet p-4 sm:grid-cols-2">
        <Field
          label="Title"
          required
          value={form.title}
          onChange={(event) => setForm({ ...form, title: event.target.value })}
        />
        <Field
          label="Due"
          type="datetime-local"
          required
          value={form.due_at}
          onChange={(event) => setForm({ ...form, due_at: event.target.value })}
        />
        <Field
          label="Department"
          required
          value={form.department}
          onChange={(event) => setForm({ ...form, department: event.target.value })}
        />
        <Field
          label="Semester"
          type="number"
          min={1}
          max={8}
          required
          value={form.semester}
          onChange={(event) => setForm({ ...form, semester: Number(event.target.value) })}
        />
        <div className="sm:col-span-2">
          <Button type="submit" variant="primary" disabled={post.isPending}>
            Post assignment
          </Button>
          {post.isSuccess && (
            <p role="status" className="mt-2 text-sm">
              Reminders sent to {post.data.reminders_created} students.
            </p>
          )}
          {post.isError && (
            <p role="alert" className="mt-2 text-error">
              {errorMessage(post.error)}
            </p>
          )}
        </div>
      </form>
      {assignments.isLoading && <SkeletonRows label="Loading assignments" />}
      {assignments.isError && <ErrorState error={assignments.error} />}
      {assignments.data?.items.length === 0 && <EmptyState title="No assignments posted" />}
      <ul className="space-y-2">
        {assignments.data?.items.map((assignment) => (
          <li key={assignment.id} className="flex items-center justify-between gap-2 border-b border-rule py-2">
            <span>
              <span className="font-medium">{assignment.title}</span>
              <span className="block text-sm text-muted">
                {assignment.department}, semester {assignment.semester}, due{" "}
                {formatDateTime(assignment.due_at)}
              </span>
            </span>
            <Button
              variant="ghost"
              aria-label={`Delete ${assignment.title}`}
              onClick={() => {
                if (window.confirm(`Delete "${assignment.title}" and its reminders?`)) {
                  remove.mutate(assignment.id);
                }
              }}
            >
              Delete
            </Button>
          </li>
        ))}
      </ul>
    </>
  );
}

export function FacultyPage() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") === "assignments" ? "assignments" : "documents";
  return (
    <>
      <PageHeader title="Faculty" />
      <Tabs
        label="Faculty sections"
        value={tab}
        options={[
          { value: "documents", label: "Documents" },
          { value: "assignments", label: "Assignments" },
        ]}
        onChange={(next) => setParams({ tab: next })}
      />
      {tab === "documents" ? (
        <>
          <UploadPanel />
          <MyDocuments />
        </>
      ) : (
        <Assignments />
      )}
    </>
  );
}
