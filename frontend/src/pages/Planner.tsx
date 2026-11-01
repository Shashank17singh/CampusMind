import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { FormEvent } from "react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { api, errorMessage } from "../api/client";
import type { Page, PlanItem, PlanSummary, StudyPlan } from "../api/types";
import { Button, EmptyState, ErrorState, Field, PageHeader, SkeletonRows } from "../components/ui";
import { formatDay, isoDay, shortTime, weekOf } from "../lib/format";

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const KIND_STYLES: Record<PlanItem["type"], string> = {
  learn: "border-l-accent",
  revise: "border-l-rule-2",
  practice: "border-l-success",
};

function PlanForm({ onCreated }: { onCreated: (plan: StudyPlan) => void }) {
  const [exams, setExams] = useState([{ subject: "", date: "", difficulty: 3 }]);
  const [hours, setHours] = useState<number[]>(WEEKDAYS.map((_, index) => (index < 5 ? 2 : 4)));

  const create = useMutation({
    mutationFn: () =>
      api
        .post<StudyPlan>("/study-plans", {
          exams: exams.map(({ subject, date }) => ({ subject, date })),
          hours_per_weekday: Object.fromEntries(hours.map((value, index) => [index, value])),
          difficulty: Object.fromEntries(exams.map((exam) => [exam.subject, exam.difficulty])),
        })
        .then((response) => response.data),
    onSuccess: onCreated,
  });

  const updateExam = (index: number, change: Partial<(typeof exams)[number]>) =>
    setExams((current) =>
      current.map((exam, position) => (position === index ? { ...exam, ...change } : exam)),
    );

  const submit = (event: FormEvent) => {
    event.preventDefault();
    create.mutate();
  };

  return (
    <form onSubmit={submit} className="space-y-4 border border-rule bg-sheet p-4">
      <fieldset>
        <legend className="mb-2 font-medium">Exams</legend>
        <div className="space-y-3">
          {exams.map((exam, index) => (
            <div key={index} className="grid gap-2 sm:grid-cols-[1fr_auto_auto]">
              <Field
                label="Subject"
                required
                value={exam.subject}
                onChange={(event) => updateExam(index, { subject: event.target.value })}
              />
              <Field
                label="Exam date"
                type="date"
                required
                value={exam.date}
                onChange={(event) => updateExam(index, { date: event.target.value })}
              />
              <Field
                label="Difficulty, 1 to 5"
                type="number"
                min={1}
                max={5}
                required
                value={exam.difficulty}
                onChange={(event) => updateExam(index, { difficulty: Number(event.target.value) })}
              />
            </div>
          ))}
        </div>
        <Button
          className="mt-2"
          onClick={() =>
            setExams((current) => [...current, { subject: "", date: "", difficulty: 3 }])
          }
        >
          Add a subject
        </Button>
      </fieldset>
      <fieldset>
        <legend className="mb-2 font-medium">Hours available</legend>
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-7">
          {WEEKDAYS.map((day, index) => (
            <Field
              key={day}
              label={day}
              type="number"
              min={0}
              max={14}
              step={0.5}
              value={hours[index]}
              onChange={(event) =>
                setHours((current) =>
                  current.map((value, position) =>
                    position === index ? Number(event.target.value) : value,
                  ),
                )
              }
            />
          ))}
        </div>
      </fieldset>
      {create.isError && (
        <p role="alert" className="text-error">
          {errorMessage(create.error)}
        </p>
      )}
      <Button type="submit" variant="primary" disabled={create.isPending}>
        {create.isPending ? "Building your plan" : "Build plan"}
      </Button>
    </form>
  );
}

function ItemCard({ item, planId }: { item: PlanItem; planId: string }) {
  const queryClient = useQueryClient();
  const key = ["study-plan", planId];
  const update = useMutation({
    mutationFn: (status: PlanItem["status"]) =>
      api.patch(`/study-plans/${planId}/items/${item.id}`, { status }),
    onMutate: async (status) => {
      await queryClient.cancelQueries({ queryKey: key });
      const previous = queryClient.getQueryData<StudyPlan>(key);
      queryClient.setQueryData<StudyPlan>(key, (plan) =>
        plan && {
          ...plan,
          items: plan.items.map((other) => (other.id === item.id ? { ...other, status } : other)),
        },
      );
      return { previous };
    },
    onError: (_error, _status, context) => queryClient.setQueryData(key, context?.previous),
  });

  return (
    <li className={`border border-l-4 border-rule bg-sheet p-2 text-sm ${KIND_STYLES[item.type]}`}>
      <p className="font-mono text-xs text-muted">
        {shortTime(item.start_time)} to {shortTime(item.end_time)}
      </p>
      <p className={item.status === "done" ? "line-through" : "font-medium"}>{item.topic}</p>
      <p className="text-muted">
        {item.subject}, {item.type}
        {item.status === "missed" && ", missed"}
      </p>
      {item.status === "pending" && (
        <div className="mt-1 flex gap-1">
          <Button variant="ghost" aria-label={`Done: ${item.topic}`} onClick={() => update.mutate("done")}>
            Done
          </Button>
          <Button variant="ghost" aria-label={`Missed: ${item.topic}`} onClick={() => update.mutate("missed")}>
            Missed
          </Button>
        </div>
      )}
    </li>
  );
}

function WeekView({ plan }: { plan: StudyPlan }) {
  const [params, setParams] = useSearchParams();
  const anchor = params.get("week") ?? plan.start_date;
  const days = useMemo(() => weekOf(new Date(`${anchor}T12:00:00`)), [anchor]);

  const move = (weeks: number) => {
    const next = new Date(`${anchor}T12:00:00`);
    next.setDate(next.getDate() + weeks * 7);
    setParams({ plan: plan.id, week: isoDay(next) });
  };

  return (
    <section aria-label="Weekly calendar">
      <div className="mb-3 flex items-center justify-between gap-2">
        <Button onClick={() => move(-1)}>Previous week</Button>
        <p className="text-sm">
          {formatDay(days[0])} to {formatDay(days[6])}
        </p>
        <Button onClick={() => move(1)}>Next week</Button>
      </div>
      <div className="grid gap-3 md:grid-cols-7">
        {days.map((day) => {
          const items = plan.items.filter((item) => item.date === day);
          return (
            <div key={day} className="min-w-0">
              <h3 className="mb-1 text-sm font-medium">{formatDay(day)}</h3>
              {items.length > 0 ? (
                <ul className="space-y-2">
                  {items.map((item) => (
                    <ItemCard key={item.id} item={item} planId={plan.id} />
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted">Nothing planned</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

async function downloadIcs(planId: string): Promise<void> {
  const response = await api.get<Blob>(`/study-plans/${planId}/export.ics`, {
    responseType: "blob",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(response.data);
  link.download = "study-plan.ics";
  link.click();
  URL.revokeObjectURL(link.href);
}

export function PlannerPage() {
  const [params, setParams] = useSearchParams();
  const queryClient = useQueryClient();
  const [creating, setCreating] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);

  const plans = useQuery({
    queryKey: ["study-plans"],
    queryFn: () =>
      api.get<Page<PlanSummary>>("/study-plans?limit=20").then((response) => response.data),
  });
  const activeId =
    params.get("plan") ?? plans.data?.items.find((plan) => plan.is_active)?.id ?? null;

  const plan = useQuery({
    queryKey: ["study-plan", activeId],
    enabled: Boolean(activeId),
    queryFn: () =>
      api.get<StudyPlan>(`/study-plans/${activeId}`).then((response) => response.data),
  });

  const replan = useMutation({
    mutationFn: () =>
      api.post<StudyPlan>(`/study-plans/${activeId}/replan`, {}).then((response) => response.data),
    onSuccess: (updated) => {
      queryClient.setQueryData(["study-plan", activeId], updated);
      setWarnings(updated.warnings);
    },
  });

  const onCreated = (created: StudyPlan) => {
    setCreating(false);
    setWarnings(created.warnings);
    queryClient.setQueryData(["study-plan", created.id], created);
    void queryClient.invalidateQueries({ queryKey: ["study-plans"] });
    setParams({ plan: created.id });
  };

  const hasPlan = Boolean(activeId);
  return (
    <>
      <PageHeader
        title="Study planner"
        actions={
          <div className="flex flex-wrap gap-2">
            {hasPlan && (
              <>
                <Button onClick={() => replan.mutate()} disabled={replan.isPending}>
                  {replan.isPending ? "Re-planning" : "Re-plan"}
                </Button>
                <Button onClick={() => void downloadIcs(activeId as string)}>Download .ics</Button>
              </>
            )}
            <Button variant="primary" onClick={() => setCreating((open) => !open)}>
              New plan
            </Button>
          </div>
        }
      />
      {warnings.length > 0 && (
        <ul className="mb-4 border border-rule-2 p-3 text-sm">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
      {replan.isError && <ErrorState error={replan.error} />}
      {(creating || (plans.data && !hasPlan)) && (
        <div className="mb-6">
          <PlanForm onCreated={onCreated} />
        </div>
      )}
      {(plans.isLoading || plan.isLoading) && <SkeletonRows rows={6} label="Loading your plan" />}
      {plans.isError && <ErrorState error={plans.error} onRetry={() => void plans.refetch()} />}
      {plan.isError && <ErrorState error={plan.error} onRetry={() => void plan.refetch()} />}
      {plan.data && <WeekView plan={plan.data} />}
      {plans.data && !hasPlan && (
        <EmptyState title="No plan yet">Add your exam dates above to build one.</EmptyState>
      )}
    </>
  );
}
