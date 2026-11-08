import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { LoginPage, RegisterPage } from "../pages/Auth";

function renderPage(page: JSX.Element, path: string) {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>{page}</MemoryRouter>
    </QueryClientProvider>,
  );
}

const renderLogin = (path = "/login") => renderPage(<LoginPage />, path);

describe("login page", () => {
  afterEach(() => vi.restoreAllMocks());

  it("offers Google when the server has it configured", async () => {
    vi.spyOn(api, "get").mockResolvedValue({ data: { google: true } });
    renderLogin();
    const link = await screen.findByRole("link", { name: "Continue with Google" });
    expect(link).toHaveAttribute("href", "/api/v1/auth/google/start");
    expect(api.get).toHaveBeenCalledWith("/auth/providers");
  });

  it("hides Google when the server does not offer it", async () => {
    vi.spyOn(api, "get").mockResolvedValue({ data: { google: false } });
    renderLogin();
    expect(await screen.findByRole("button", { name: "Log in" })).toBeInTheDocument();
    await vi.waitFor(() => expect(api.get).toHaveBeenCalled());
    expect(screen.queryByRole("link", { name: "Continue with Google" })).toBeNull();
  });

  it("explains why Google sign-in was refused", async () => {
    vi.spyOn(api, "get").mockResolvedValue({ data: { google: true } });
    renderLogin("/login?error=domain");
    expect(await screen.findByRole("alert")).toHaveTextContent("Use your college Google account.");
  });

  it("in roster mode, points students to their department instead of sign-up", async () => {
    vi.spyOn(api, "get").mockResolvedValue({ data: { google: true, registration: "roster" } });
    renderLogin();
    expect(await screen.findByText(/Accounts are added by your department/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Create an account" })).toBeNull();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
  });

  it("explains a Google account that is not on the roster", async () => {
    vi.spyOn(api, "get").mockResolvedValue({ data: { google: true, registration: "roster" } });
    renderLogin("/login?error=roster");
    expect(await screen.findByRole("alert")).toHaveTextContent("not on your department's list");
  });

  it("in roster mode, the register page offers to set a password instead", async () => {
    vi.spyOn(api, "get").mockResolvedValue({ data: { google: false, registration: "roster" } });
    renderPage(<RegisterPage />, "/register");
    const link = await screen.findByRole("link", { name: "Set my password" });
    expect(link).toHaveAttribute("href", "/forgot-password");
    expect(screen.queryByLabelText("Password")).toBeNull();
  });
});
