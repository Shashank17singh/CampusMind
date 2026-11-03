import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { FormEvent } from "react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import { api, errorMessage } from "../api/client";
import type { Page, ParseResult, Reminder, ReminderView } from "../api/types";
import {
  Button,
  EmptyState,
  ErrorState,
  PageHeader,
  SkeletonRows,
  Tabs,
} from "../components/ui";
import { formatDateTime } from "../lib/format";

const VIEWS: { value: ReminderView; label: string }[] = [
  { value: "upcoming", label: "Upcoming" },
  { value: "overdue", label: "Overdue" },
  { value: "done", label: "Done" },
];
const EMPTY: Record<ReminderView, string> = {
  upcoming: "Nothing coming up. Add one above.",
  overdue: "Nothing overdue.",
  done: "Finished reminders appear here.",
};

function QuickAdd() {
  const [text, setText] = useState("");
  const queryClient = useQueryClient();
  const add = useMutation({
    mutationFn: () =>
      api.post<ParseResult>("/reminders/parse", { text }).then((response) => response.data),
    onSuccess: (result) => {
      if (result.reminder) {
        setText("");
        void queryClient.invalidateQueries({ queryKey: ["reminders"] });
      }
    },
  });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (text.trim()) add.mutate();
  };

  return (
    <form onSubmit={submit} className="mb-6">
      <div className="flex gap-2">
        <input
          aria-label="New reminder"
          className="min-h-10 min-w-0 flex-1 field px-3"
          placeholder="Submit the DBMS assignment on Friday 5pm"
          value={text}
          onChange={(event) => setText(event.target.value)}
        />
        <Button type="submit" variant="primary" disabled={add.isPending || !text.trim()}>
          Add
        </Button>
      </div>
      {add.data?.clarification && (
        <p role="status" className="mt-2">
          {add.data.clarification}
        </p>
      )}
      {add.isError && (
        <p role="alert" className="mt-2 text-error">
          {errorMessage(add.error)}
        </p>
      )}
    </form>
  );
}

function ReminderRow({ reminder, view }: { reminder: Reminder; view: ReminderView }) {
  const queryClient = useQueryClient();
  const key = ["reminders", view];

  const removeFromList = async () => {
    await queryClient.cancelQueries({ queryKey: key });
    const previous = queryClient.getQueryData<Page<Reminder>>(key);
    queryClient.setQueryData<Page<Reminder>>(key, (page) =>
      page && {
        total: page.total - 1,
        items: page.items.filter((item) => item.id !== reminder.id),
      },
    );
    return { previous };
  };
  const restore = (context?: { previous?: Page<Reminder> }) =>
    queryClient.setQueryData(key, context?.previous);
  const refresh = () => queryClient.invalidateQueries({ queryKey: ["reminders"] });

  const done = useMutation({
    mutationFn: () => api.post(`/reminders/${reminder.id}/done`),
    onMutate: removeFromList,
    onError: (_error, _input, context) => restore(context),
    onSettled: refresh,
  });
  const remove = useMutation({
    mutationFn: () => api.delete(`/reminders/${reminder.id}`),
    onMutate: removeFromList,
    onError: (_error, _input, context) => restore(context),
    onSettled: refresh,
  });
  const snooze = useMutation({
    mutationFn: () => api.post(`/reminders/${reminder.id}/snooze`, { minutes: 60 }),
    onSettled: refresh,
  });

  return (
    <tr className="border-b border-rule">
      <td className="py-2 pr-3">
        <span className="font-medium">{reminder.title}</span>
        {reminder.recurrence && (
          <span className="ml-2 text-sm text-muted">repeats {reminder.recurrence}</span>
        )}
        <span className="block text-sm text-muted sm:hidden">
          {formatDateTime(reminder.due_at)}
        </span>
      </td>
      <td className="hidden py-2 pr-3 text-sm tabular-nums sm:table-cell">
        {formatDateTime(reminder.due_at)}
      </td>
      <td className="py-2 text-right">
        <div className="flex flex-wrap justify-end gap-1">
          {view !== "done" && (
            <>
              <Button variant="ghost" aria-label={`Done: ${reminder.title}`} onClick={() => done.mutate()}>
                Done
              </Button>
              <Button variant="ghost" aria-label={`Snooze ${reminder.title} one hour`} onClick={() => snooze.mutate()}>
                Snooze
              </Button>
            </>
          )}
          <Button variant="ghost" aria-label={`Delete ${reminder.title}`} onClick={() => remove.mutate()}>
            Delete
          </Button>
        </div>
      </td>
    </tr>
  );
}

export function RemindersPage() {
  const [params, setParams] = useSearchParams();
  const view = (params.get("view") as ReminderView | null) ?? "upcoming";
  const reminders = useQuery({
    queryKey: ["reminders", view],
    queryFn: () =>
      api
        .get<Page<Reminder>>("/reminders", { params: { view } })
        .then((response) => response.data),
  });

  return (
    <>
      <PageHeader title="Reminders" />
      <QuickAdd />
      <Tabs
        label="Reminder views"
        value={view}
        options={VIEWS}
        onChange={(next) => setParams({ view: next })}
      />
      {reminders.isLoading && <SkeletonRows label="Loading reminders" />}
      {reminders.isError && (
        <ErrorState error={reminders.error} onRetry={() => void reminders.refetch()} />
      )}
      {reminders.data?.items.length === 0 && <EmptyState title={EMPTY[view]} />}
      {reminders.data && reminders.data.items.length > 0 && (
        <table className="w-full">
          <caption className="sr-only">{view} reminders</caption>
          <thead>
            <tr className="border-b border-rule text-left text-sm">
              <th scope="col" className="py-2 font-medium">
                Reminder
              </th>
              <th scope="col" className="hidden py-2 font-medium sm:table-cell">
                Due
              </th>
              <th scope="col" className="py-2 text-right font-medium">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {reminders.data.items.map((reminder) => (
              <ReminderRow key={reminder.id} reminder={reminder} view={view} />
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
