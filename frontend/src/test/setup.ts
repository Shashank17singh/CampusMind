import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { useAuth } from "../stores/auth";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useAuth.setState({ accessToken: null, user: null, ready: false });
});
