import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { NavLink, Navigate, Outlet, useLocation, useNavigate } from "react-router-dom";

import { api } from "../api/client";
import type { Role } from "../api/types";
import { useNotifications } from "../hooks/useNotifications";
import { useAuth } from "../stores/auth";
import { useUi } from "../stores/ui";
import { Button, SkeletonRows } from "./ui";

const LINKS: { to: string; label: string; roles?: Role[] }[] = [
  { to: "/chat", label: "Chat" },
  { to: "/planner", label: "Planner" },
  { to: "/reminders", label: "Reminders" },
  { to: "/summarizer", label: "Summarizer" },
  { to: "/faculty", label: "Faculty", roles: ["faculty", "admin"] },
  { to: "/admin", label: "Admin", roles: ["admin"] },
];

function Toasts() {
  const toasts = useUi((state) => state.toasts);
  const dismiss = useUi((state) => state.dismissToast);
  return (
    <div
      aria-live="polite"
      className="fixed bottom-4 right-4 z-toast flex w-[min(22rem,calc(100vw-2rem))] flex-col gap-2"
    >
      {toasts.map((toast) => (
        <div key={toast.id} className="panel p-4">
          <div className="flex items-start justify-between gap-2">
            <p className="font-medium">{toast.title}</p>
            <button
              type="button"
              aria-label={`Dismiss ${toast.title}`}
              className="min-h-11 min-w-11 rounded-full text-muted hover:bg-paper-3"
              onClick={() => dismiss(toast.id)}
            >
              ×
            </button>
          </div>
          {toast.body && <p className="mt-1 text-sm text-muted">{toast.body}</p>}
        </div>
      ))}
    </div>
  );
}

export function Shell() {
  const user = useAuth((state) => state.user);
  const clear = useAuth((state) => state.clear);
  const theme = useUi((state) => state.theme);
  const toggleTheme = useUi((state) => state.toggleTheme);
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  useNotifications();

  useEffect(() => setMenuOpen(false), [location.pathname]);

  const logOut = async () => {
    await api.post("/auth/logout").catch(() => undefined);
    clear();
    queryClient.clear();
    navigate("/login");
  };

  const links = LINKS.filter((link) => !link.roles || (user && link.roles.includes(user.role)));

  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:p-2">
        Skip to content
      </a>
      <header className="sticky top-0 z-sticky border-b border-rule bg-paper">
        <div className="mx-auto flex h-16 max-w-content items-center gap-6 px-4">
          <NavLink to="/chat" className="font-display text-lg font-medium tracking-tight">
            CampusMind
          </NavLink>
          <nav aria-label="Main" className="hidden gap-1 rounded-full bg-paper-2 p-1 md:flex">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) =>
                  `inline-flex min-h-9 items-center rounded-full px-4 text-sm transition-colors ${
                    isActive ? "control font-medium text-ink" : "text-muted hover:text-ink"
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-1">
            <Button variant="ghost" onClick={toggleTheme}>
              {theme === "dark" ? "Light" : "Dark"}
            </Button>
            <Button variant="ghost" className="hidden md:inline-flex" onClick={logOut}>
              Log out
            </Button>
            <Button
              variant="ghost"
              className="md:hidden"
              aria-expanded={menuOpen}
              aria-controls="mobile-menu"
              onClick={() => setMenuOpen((open) => !open)}
            >
              Menu
            </Button>
          </div>
        </div>
        {menuOpen && (
          <nav id="mobile-menu" aria-label="Mobile" className="border-t border-rule p-2 md:hidden">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) => `block rounded px-3 py-3 ${isActive ? "bg-paper-3 font-medium" : ""}`}
              >
                {link.label}
              </NavLink>
            ))}
            <button type="button" className="block w-full px-3 py-3 text-left" onClick={logOut}>
              Log out
            </button>
          </nav>
        )}
      </header>
      <main id="main" className="mx-auto w-full max-w-content flex-1 px-4 py-8">
        <Outlet />
      </main>
      <Toasts />
    </div>
  );
}

export function RequireAuth({ roles }: { roles?: Role[] }) {
  const user = useAuth((state) => state.user);
  const ready = useAuth((state) => state.ready);
  const location = useLocation();

  if (!ready) {
    return (
      <div className="mx-auto max-w-content p-8">
        <SkeletonRows label="Checking your session" />
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  if (roles && !roles.includes(user.role)) return <Navigate to="/chat" replace />;
  return <Outlet />;
}
