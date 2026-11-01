import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from "react";
import { Component, useId } from "react";

import { errorMessage } from "../api/client";

type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-accent-ink hover:bg-accent-2",
  secondary: "border border-rule-2 text-ink hover:bg-paper-2",
  ghost: "text-ink hover:bg-paper-2",
  danger: "border border-error text-error hover:bg-paper-2",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

export function Button({ variant = "secondary", className = "", type, ...rest }: ButtonProps) {
  return (
    <button
      type={type ?? "button"}
      className={`inline-flex min-h-10 items-center justify-center gap-2 rounded px-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${className}`}
      {...rest}
    />
  );
}

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  hint?: string;
  error?: string | null;
}

export function Field({ label, hint, error, className = "", ...rest }: FieldProps) {
  const id = useId();
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;
  return (
    <div className={className}>
      <label htmlFor={id} className="mb-1 block text-sm font-medium">
        {label}
      </label>
      <input
        id={id}
        aria-invalid={error ? true : undefined}
        aria-describedby={describedBy}
        className="min-h-10 w-full rounded border border-rule-2 bg-sheet px-3 text-ink"
        {...rest}
      />
      {hint && !error && (
        <p id={`${id}-hint`} className="mt-1 text-sm text-muted">
          {hint}
        </p>
      )}
      {error && (
        <p id={`${id}-error`} className="mt-1 text-sm text-error">
          {error}
        </p>
      )}
    </div>
  );
}

export function SkeletonRows({ rows = 4, label = "Loading" }: { rows?: number; label?: string }) {
  return (
    <div role="status" aria-label={label} className="space-y-2">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="skeleton h-10 w-full" />
      ))}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="border border-dashed border-rule-2 p-8 text-center">
      <p className="font-medium">{title}</p>
      {children && <div className="mt-2 text-sm text-muted">{children}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div role="alert" className="border border-error p-4">
      <p className="text-error">{errorMessage(error)}</p>
      {onRetry && (
        <Button className="mt-3" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

export function PageHeader({ title, actions }: { title: string; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {actions}
    </div>
  );
}

interface TabsProps<T extends string> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}

export function Tabs<T extends string>({ label, value, options, onChange }: TabsProps<T>) {
  return (
    <div role="tablist" aria-label={label} className="mb-4 flex gap-1 border-b border-rule">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="tab"
          aria-selected={option.value === value}
          onClick={() => onChange(option.value)}
          className={`min-h-10 border-b-2 px-3 text-sm ${
            option.value === value
              ? "border-accent font-medium text-ink"
              : "border-transparent text-muted hover:text-ink"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

interface BoundaryState {
  failed: boolean;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, BoundaryState> {
  state: BoundaryState = { failed: false };

  static getDerivedStateFromError(): BoundaryState {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <main className="mx-auto max-w-prose p-8">
          <h1 className="text-2xl font-semibold">This page hit a problem</h1>
          <p className="mt-2 text-muted">Reloading usually fixes it.</p>
          <Button className="mt-4" variant="primary" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </main>
      );
    }
    return this.props.children;
  }
}
