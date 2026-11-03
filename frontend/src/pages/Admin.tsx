import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api, errorMessage } from "../api/client";
import type { DocumentRecord, Overview, Page, QuestionCluster, User } from "../api/types";
import {
  Button,
  EmptyState,
  ErrorState,
  PageHeader,
  SkeletonRows,
  Tabs,
} from "../components/ui";

type Tab = "analytics" | "gaps" | "users" | "documents";

const TABS: { value: Tab; label: string }[] = [
  { value: "analytics", label: "Analytics" },
  { value: "gaps", label: "Knowledge gaps" },
  { value: "users", label: "Users" },
  { value: "documents", label: "Documents" },
];

const AXIS = { fontSize: 12, fill: "var(--color-muted)" };

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-t border-rule pt-3">
      <p className="text-sm text-muted">{label}</p>
      <p className="mt-1 font-display text-3xl tabular-nums">{value}</p>
    </div>
  );
}

function ChartBox({ title, children }: { title: string; children: React.ReactElement }) {
  return (
    <figure className="min-w-0">
      <figcaption className="mb-3 text-sm font-medium text-muted">{title}</figcaption>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

function Analytics() {
  const overview = useQuery({
    queryKey: ["analytics", "overview"],
    queryFn: () =>
      api.get<Overview>("/admin/analytics/overview").then((response) => response.data),
  });
  const intents = useQuery({
    queryKey: ["analytics", "intents"],
    queryFn: () =>
      api
        .get<{ intent: string; count: number }[]>("/admin/analytics/intents")
        .then((response) => response.data),
  });
  const documents = useQuery({
    queryKey: ["analytics", "documents"],
    queryFn: () =>
      api
        .get<{ title: string; citations: number }[]>("/admin/analytics/top-documents")
        .then((response) => response.data),
  });

  if (overview.isLoading) return <SkeletonRows rows={6} label="Loading analytics" />;
  if (overview.isError) {
    return <ErrorState error={overview.error} onRetry={() => void overview.refetch()} />;
  }
  const data = overview.data;
  if (!data || data.daily.length === 0) {
    return <EmptyState title="No activity in the last 30 days" />;
  }
  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Figure
          label="Questions"
          value={String(data.daily.reduce((sum, day) => sum + day.questions, 0))}
        />
        <Figure
          label="Average latency"
          value={data.avg_latency_ms === null ? "None yet" : `${Math.round(data.avg_latency_ms)} ms`}
        />
        <Figure
          label="Helpful ratings"
          value={data.feedback_ratio === null ? "None yet" : `${Math.round(data.feedback_ratio * 100)}%`}
        />
        <Figure label="Rated unhelpful" value={String(data.thumbs_down)} />
      </div>
      <div className="grid gap-8 md:grid-cols-2">
        <ChartBox title="Questions per day">
          <LineChart data={data.daily}>
            <CartesianGrid stroke="var(--color-rule)" vertical={false} />
            <XAxis dataKey="day" tick={AXIS} />
            <YAxis allowDecimals={false} tick={AXIS} width={32} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="questions"
              stroke="var(--color-accent)"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartBox>
        <ChartBox title="Daily active users">
          <LineChart data={data.daily}>
            <CartesianGrid stroke="var(--color-rule)" vertical={false} />
            <XAxis dataKey="day" tick={AXIS} />
            <YAxis allowDecimals={false} tick={AXIS} width={32} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="active_users"
              stroke="var(--color-ink)"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartBox>
        {intents.data && (
          <ChartBox title="Intents">
            <BarChart data={intents.data}>
              <CartesianGrid stroke="var(--color-rule)" vertical={false} />
              <XAxis dataKey="intent" tick={AXIS} />
              <YAxis allowDecimals={false} tick={AXIS} width={32} />
              <Tooltip />
              <Bar dataKey="count" fill="var(--color-accent)" isAnimationActive={false} />
            </BarChart>
          </ChartBox>
        )}
        {documents.data && documents.data.length > 0 && (
          <section>
            <h2 className="mb-2 font-medium">Most cited documents</h2>
            <ol className="space-y-1 text-sm">
              {documents.data.slice(0, 8).map((document) => (
                <li key={document.title} className="flex justify-between gap-2 border-b border-rule py-1">
                  <span className="min-w-0 truncate">{document.title}</span>
                  <span className="tabular-nums">{document.citations}</span>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </div>
  );
}

function KnowledgeGaps() {
  const gaps = useQuery({
    queryKey: ["analytics", "gaps"],
    queryFn: () =>
      api
        .get<QuestionCluster[]>("/admin/analytics/knowledge-gaps")
        .then((response) => response.data),
  });
  if (gaps.isLoading) return <SkeletonRows label="Loading knowledge gaps" />;
  if (gaps.isError) return <ErrorState error={gaps.error} onRetry={() => void gaps.refetch()} />;
  if (!gaps.data?.length) {
    return <EmptyState title="No gaps found">Every recent question had a good answer.</EmptyState>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Topics students asked about without a good answer</caption>
        <thead>
          <tr className="border-b border-rule text-left">
            <th scope="col" className="py-2 font-medium">Topic</th>
            <th scope="col" className="py-2 font-medium">Times asked</th>
            <th scope="col" className="py-2 font-medium">Examples</th>
          </tr>
        </thead>
        <tbody>
          {gaps.data.map((gap) => (
            <tr key={gap.topic} className="border-b border-rule align-top">
              <td className="py-2 pr-3">{gap.topic}</td>
              <td className="py-2 pr-3 tabular-nums">{gap.count}</td>
              <td className="py-2 text-muted">{gap.examples.join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Users() {
  const queryClient = useQueryClient();
  const [query, setQuery] = useState("");
  const fileId = useId();
  const users = useQuery({
    queryKey: ["admin-users", query],
    queryFn: () =>
      api
        .get<Page<User>>("/admin/users", { params: { q: query || undefined } })
        .then((response) => response.data),
  });
  const importCsv = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api
        .post<{ created: number; skipped: number; errors: string[] }>("/admin/users/import", form)
        .then((response) => response.data);
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin-users"] }),
  });
  const setRole = useMutation({
    mutationFn: ({ id, role }: { id: string; role: string }) =>
      api.patch(`/admin/users/${id}`, { role }),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["admin-users"] }),
  });

  return (
    <>
      <div className="mb-4 flex flex-wrap items-end gap-4">
        <input
          type="search"
          aria-label="Search users"
          placeholder="Search by name or email"
          className="min-h-10 min-w-0 flex-1 field px-3"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <div>
          <label htmlFor={fileId} className="mb-1 block text-sm font-medium">
            Import students from CSV
          </label>
          <input
            id={fileId}
            type="file"
            accept=".csv"
            className="text-sm"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) importCsv.mutate(file);
              event.target.value = "";
            }}
          />
        </div>
      </div>
      {importCsv.data && (
        <div role="status" className="mb-4 border border-rule-2 p-3 text-sm">
          <p>
            {importCsv.data.created} created, {importCsv.data.skipped} skipped.
          </p>
          {importCsv.data.errors.map((error) => (
            <p key={error} className="text-error">
              {error}
            </p>
          ))}
        </div>
      )}
      {importCsv.isError && (
        <p role="alert" className="mb-4 text-error">
          {errorMessage(importCsv.error)}
        </p>
      )}
      {users.isLoading && <SkeletonRows label="Loading users" />}
      {users.isError && <ErrorState error={users.error} onRetry={() => void users.refetch()} />}
      {users.data?.items.length === 0 && <EmptyState title="No users match" />}
      {users.data && users.data.items.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <caption className="sr-only">Users</caption>
            <thead>
              <tr className="border-b border-rule text-left">
                <th scope="col" className="py-2 font-medium">Name</th>
                <th scope="col" className="py-2 font-medium">Email</th>
                <th scope="col" className="py-2 font-medium">Department</th>
                <th scope="col" className="py-2 font-medium">Role</th>
              </tr>
            </thead>
            <tbody>
              {users.data.items.map((user) => (
                <tr key={user.id} className="border-b border-rule">
                  <td className="py-2 pr-3">{user.full_name}</td>
                  <td className="py-2 pr-3">{user.email}</td>
                  <td className="py-2 pr-3">{user.department ?? "None"}</td>
                  <td className="py-2">
                    <select
                      aria-label={`Role for ${user.full_name}`}
                      className="min-h-10 field px-2"
                      value={user.role}
                      onChange={(event) => setRole.mutate({ id: user.id, role: event.target.value })}
                    >
                      <option value="student">Student</option>
                      <option value="faculty">Faculty</option>
                      <option value="admin">Admin</option>
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function Documents() {
  const queryClient = useQueryClient();
  const documents = useQuery({
    queryKey: ["documents", "all"],
    queryFn: () =>
      api
        .get<Page<DocumentRecord>>("/documents", { params: { limit: 100 } })
        .then((response) => response.data),
  });
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["documents"] });
  const remove = useMutation({
    mutationFn: (id: string) => api.delete(`/documents/${id}`),
    onSettled: refresh,
  });
  const reingest = useMutation({
    mutationFn: (id: string) => api.post(`/documents/${id}/reingest`),
    onSettled: refresh,
  });

  if (documents.isLoading) return <SkeletonRows label="Loading documents" />;
  if (documents.isError) {
    return <ErrorState error={documents.error} onRetry={() => void documents.refetch()} />;
  }
  if (!documents.data?.items.length) return <EmptyState title="No documents yet" />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">All documents</caption>
        <thead>
          <tr className="border-b border-rule text-left">
            <th scope="col" className="py-2 font-medium">Title</th>
            <th scope="col" className="py-2 font-medium">Department</th>
            <th scope="col" className="py-2 font-medium">Status</th>
            <th scope="col" className="py-2 text-right font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.data.items.map((document) => (
            <tr key={document.id} className="border-b border-rule">
              <td className="py-2 pr-3">
                {document.title}
                {document.superseded_by_id && <span className="ml-2 text-muted">superseded</span>}
              </td>
              <td className="py-2 pr-3">{document.department ?? "College-wide"}</td>
              <td className="py-2 pr-3">{document.status}</td>
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

export function AdminPage() {
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as Tab | null) ?? "analytics";
  return (
    <>
      <PageHeader title="Admin" />
      <Tabs
        label="Admin sections"
        value={tab}
        options={TABS}
        onChange={(next) => setParams({ tab: next })}
      />
      {tab === "analytics" && <Analytics />}
      {tab === "gaps" && <KnowledgeGaps />}
      {tab === "users" && <Users />}
      {tab === "documents" && <Documents />}
    </>
  );
}
