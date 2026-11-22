import { useMutation, useQuery } from "@tanstack/react-query";
import type { FormEvent, ReactNode } from "react";
import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";

import { API_BASE, api, errorMessage } from "../api/client";
import type { User } from "../api/types";
import { Button, Field, buttonClasses } from "../components/ui";
import { useAuth } from "../stores/auth";

function AuthCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <main className="flex min-h-dvh items-center justify-center p-4">
      <div className="panel w-full max-w-sm p-8">
        <Link to="/chat" className="mb-8 block font-display text-lg font-medium tracking-tight hover:opacity-80">
          CampusMind
        </Link>
        <h1 className="page-title mb-6">{title}</h1>
        {children}
      </div>
    </main>
  );
}

function Notice({ tone, children }: { tone: "error" | "info"; children: ReactNode }) {
  return (
    <p role={tone === "error" ? "alert" : "status"} className={tone === "error" ? "text-error" : ""}>
      {children}
    </p>
  );
}

interface Providers {
  google: boolean;
  registration: "open" | "roster";
}

/** Which sign-in options the server offers, and whether anyone may sign up. */
function useProviders() {
  return useQuery({
    queryKey: ["auth-providers"],
    queryFn: () => api.get<Providers>("/auth/providers").then((response) => response.data),
    staleTime: Infinity,
    retry: false,
  });
}

const ROSTER_HELP =
  "Accounts are added by your department. If you are on the list, use Forgot password to set your password.";

const GOOGLE_ERRORS: Record<string, string> = {
  domain: "Use your college Google account.",
  roster: "This Google account is not on your department's list. Use the email address the department has for you.",
  cancelled: "Google sign-in was cancelled.",
  google: "Google sign-in did not work. Try again, or log in with your password.",
};

export function LoginPage() {
  const setSession = useAuth((state) => state.setSession);
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [socialNotice, setSocialNotice] = useState<string | null>(null);
  const [params] = useSearchParams();
  const googleError = params.get("error");

  const providers = useProviders();
  const roster = providers.data?.registration === "roster";

  const login = useMutation({
    mutationFn: () =>
      api
        .post<{ access_token: string; user: User }>("/auth/login", { email, password })
        .then((response) => response.data),
    onSuccess: (data) => {
      setSession(data.access_token, data.user);
      const from = (location.state as { from?: string } | null)?.from;
      navigate(from ?? "/chat", { replace: true });
    },
  });

  const submit = (event: FormEvent) => {
    event.preventDefault();
    login.mutate();
  };

  return (
    <AuthCard title="Log in">
      {googleError && (
        <div className="mb-4">
          <Notice tone="error">{GOOGLE_ERRORS[googleError] ?? GOOGLE_ERRORS.google}</Notice>
        </div>
      )}
      <form onSubmit={submit} className="space-y-4">
        <Field
          label={roster ? "Email" : "College email"}
          type="email"
          autoComplete="email"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <Field
          label="Password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {login.isError && <Notice tone="error">{errorMessage(login.error)}</Notice>}
        <Button type="submit" variant="primary" className="w-full" disabled={login.isPending}>
          {login.isPending ? "Logging in" : "Log in"}
        </Button>
      </form>
      <div className="mt-2">
        <button
          type="button"
          onClick={() => {
            setEmail("student@college.edu");
            setPassword("DemoPassword123!");
          }}
          className={buttonClasses("ghost", "w-full text-xs text-muted hover:text-ink border border-dashed border-rule")}
        >
          Use demo credentials (student@college.edu)
        </button>
      </div>
      <div className="mt-3 space-y-2">
        {providers.data?.google ? (
          <a href={`${API_BASE}/auth/google/start`} className={buttonClasses("secondary", "w-full")}>
            Continue with Google
          </a>
        ) : (
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setSocialNotice("GitHub sign-in is not available yet. Please log in with your college email.")}
              className={buttonClasses("secondary", "flex-1")}
            >
              GitHub
            </button>
            <button
              type="button"
              onClick={() => setSocialNotice("Google sign-in is not available yet. Please log in with your college email.")}
              className={buttonClasses("secondary", "flex-1")}
            >
              Google
            </button>
          </div>
        )}
        {socialNotice && (
          <p className="rounded border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-center text-xs text-amber-500">
            {socialNotice}
          </p>
        )}
      </div>
      {roster && <p className="mt-4 text-sm">{ROSTER_HELP}</p>}
      <div className="mt-4 flex justify-between text-sm">
        {!roster && (
          <Link to="/register" className="underline">
            Create an account
          </Link>
        )}
        <Link to="/forgot-password" className="underline">
          Forgot password
        </Link>
      </div>
    </AuthCard>
  );
}

export function RegisterPage() {
  const providers = useProviders();
  const [form, setForm] = useState({ full_name: "", email: "", password: "", department: "" });
  const [semester, setSemester] = useState("");
  const [socialNotice, setSocialNotice] = useState<string | null>(null);

  const register = useMutation({
    mutationFn: () =>
      api.post("/auth/register", {
        ...form,
        department: form.department || null,
        semester: semester ? Number(semester) : null,
      }),
  });

  const update = (key: keyof typeof form) => (event: { target: { value: string } }) =>
    setForm((current) => ({ ...current, [key]: event.target.value }));

  if (providers.data?.registration === "roster") {
    return (
      <AuthCard title="Create an account">
        <Notice tone="info">{ROSTER_HELP}</Notice>
        <Link to="/forgot-password" className={buttonClasses("primary", "mt-6 w-full")}>
          Set my password
        </Link>
      </AuthCard>
    );
  }

  if (register.isSuccess) {
    return (
      <AuthCard title="Check your email">
        <Notice tone="info">
          If that address can be registered, a verification link is on its way.
        </Notice>
      </AuthCard>
    );
  }

  return (
    <AuthCard title="Create your account">
      <form
        className="space-y-4"
        onSubmit={(event) => {
          event.preventDefault();
          register.mutate();
        }}
      >
        <Field label="Full name" required value={form.full_name} onChange={update("full_name")} />
        <Field
          label="College email"
          type="email"
          required
          autoComplete="email"
          value={form.email}
          onChange={update("email")}
        />
        <Field
          label="Password"
          type="password"
          required
          minLength={10}
          autoComplete="new-password"
          hint="At least 10 characters"
          value={form.password}
          onChange={update("password")}
        />
        <div className="grid grid-cols-2 gap-3">
          <Field label="Department" value={form.department} onChange={update("department")} />
          <Field
            label="Semester"
            type="number"
            min={1}
            max={8}
            value={semester}
            onChange={(event) => setSemester(event.target.value)}
          />
        </div>
        {register.isError && <Notice tone="error">{errorMessage(register.error)}</Notice>}
        <Button type="submit" variant="primary" className="w-full" disabled={register.isPending}>
          Create account
        </Button>
      </form>
      <div className="mt-3 space-y-2">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setSocialNotice("GitHub registration is not available yet. Please use your college email.")}
            className={buttonClasses("secondary", "flex-1")}
          >
            GitHub
          </button>
          <button
            type="button"
            onClick={() => setSocialNotice("Google registration is not available yet. Please use your college email.")}
            className={buttonClasses("secondary", "flex-1")}
          >
            Google
          </button>
        </div>
        {socialNotice && (
          <p className="rounded border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-center text-xs text-amber-500">
            {socialNotice}
          </p>
        )}
      </div>
      <p className="mt-4 text-sm">
        <Link to="/login" className="underline">
          Back to log in
        </Link>
      </p>
    </AuthCard>
  );
}

export function VerifyEmailPage() {
  const [params] = useSearchParams();
  const token = params.get("token");
  const verify = useMutation({ mutationFn: () => api.post("/auth/verify-email", { token }) });
  const { mutate } = verify;

  useEffect(() => {
    if (token) mutate();
  }, [token, mutate]);

  return (
    <AuthCard title="Verify your email">
      {!token && <Notice tone="error">This link has no token.</Notice>}
      {verify.isPending && <Notice tone="info">Checking your link</Notice>}
      {verify.isError && <Notice tone="error">{errorMessage(verify.error)}</Notice>}
      {verify.isSuccess && <Notice tone="info">Your email address is verified.</Notice>}
      <p className="mt-4 text-sm">
        <Link to="/login" className="underline">
          Go to log in
        </Link>
      </p>
    </AuthCard>
  );
}

export function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const request = useMutation({ mutationFn: () => api.post("/auth/forgot-password", { email }) });

  return (
    <AuthCard title="Reset your password">
      {request.isSuccess ? (
        <Notice tone="info">If that account exists, a reset link is on its way.</Notice>
      ) : (
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            request.mutate();
          }}
        >
          <Field
            label="College email"
            type="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          {request.isError && <Notice tone="error">{errorMessage(request.error)}</Notice>}
          <Button type="submit" variant="primary" className="w-full" disabled={request.isPending}>
            Send reset link
          </Button>
        </form>
      )}
      <p className="mt-4 text-sm">
        <Link to="/login" className="underline">
          Back to log in
        </Link>
      </p>
    </AuthCard>
  );
}

export function ResetPasswordPage() {
  const [params] = useSearchParams();
  const [password, setPassword] = useState("");
  const reset = useMutation({
    mutationFn: () =>
      api.post("/auth/reset-password", { token: params.get("token"), new_password: password }),
  });

  return (
    <AuthCard title="Choose a new password">
      {reset.isSuccess ? (
        <Notice tone="info">Your password is changed. Every other session is signed out.</Notice>
      ) : (
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            reset.mutate();
          }}
        >
          <Field
            label="New password"
            type="password"
            required
            minLength={10}
            autoComplete="new-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          {reset.isError && <Notice tone="error">{errorMessage(reset.error)}</Notice>}
          <Button type="submit" variant="primary" className="w-full" disabled={reset.isPending}>
            Change password
          </Button>
        </form>
      )}
      <p className="mt-4 text-sm">
        <Link to="/login" className="underline">
          Go to log in
        </Link>
      </p>
    </AuthCard>
  );
}
