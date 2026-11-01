import { Suspense, lazy, useEffect } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { api, refreshAccessToken } from "./api/client";
import type { User } from "./api/types";
import { RequireAuth, Shell } from "./components/Shell";
import { SkeletonRows } from "./components/ui";
import {
  ForgotPasswordPage,
  LoginPage,
  RegisterPage,
  ResetPasswordPage,
  VerifyEmailPage,
} from "./pages/Auth";
import { useAuth } from "./stores/auth";

// Pages load on demand, so a student never downloads the admin charts.
const AdminPage = lazy(() => import("./pages/Admin").then((m) => ({ default: m.AdminPage })));
const ChatPage = lazy(() => import("./pages/Chat").then((m) => ({ default: m.ChatPage })));
const FacultyPage = lazy(() => import("./pages/Faculty").then((m) => ({ default: m.FacultyPage })));
const PlannerPage = lazy(() => import("./pages/Planner").then((m) => ({ default: m.PlannerPage })));
const RemindersPage = lazy(() =>
  import("./pages/Reminders").then((m) => ({ default: m.RemindersPage })),
);
const SummarizerPage = lazy(() =>
  import("./pages/Summarizer").then((m) => ({ default: m.SummarizerPage })),
);

/** Restore the session from the refresh cookie when the app loads. */
function useSessionRestore(): void {
  useEffect(() => {
    const { setUser, setReady, clear } = useAuth.getState();
    refreshAccessToken()
      .then(() => api.get<User>("/users/me"))
      .then((response) => setUser(response.data))
      .catch(() => clear())
      .finally(setReady);
  }, []);
}

export function App() {
  useSessionRestore();
  return (
    <Suspense fallback={<SkeletonRows label="Loading page" />}>
      <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/verify-email" element={<VerifyEmailPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<Shell />}>
          <Route path="/chat/:conversationId?" element={<ChatPage />} />
          <Route path="/planner" element={<PlannerPage />} />
          <Route path="/reminders" element={<RemindersPage />} />
          <Route path="/summarizer" element={<SummarizerPage />} />
          <Route element={<RequireAuth roles={["faculty", "admin"]} />}>
            <Route path="/faculty" element={<FacultyPage />} />
          </Route>
          <Route element={<RequireAuth roles={["admin"]} />}>
            <Route path="/admin" element={<AdminPage />} />
          </Route>
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </Suspense>
  );
}
