import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { LoginPage } from "../pages/Auth";

function renderLogin(path = "/login") {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <LoginPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

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
});
