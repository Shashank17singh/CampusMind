import axios from "axios";
import { describe, expect, it, vi } from "vitest";

import { errorMessage, refreshAccessToken } from "../api/client";
import { useAuth } from "../stores/auth";

describe("token refresh", () => {
  it("shares one request between simultaneous callers", async () => {
    const post = vi.spyOn(axios, "post").mockResolvedValue({ data: { access_token: "new" } });
    const results = await Promise.all([
      refreshAccessToken(),
      refreshAccessToken(),
      refreshAccessToken(),
    ]);
    expect(post).toHaveBeenCalledTimes(1);
    expect(results).toEqual(["new", "new", "new"]);
    expect(useAuth.getState().accessToken).toBe("new");
  });

  it("makes a fresh request after the last one finished", async () => {
    const post = vi.spyOn(axios, "post").mockResolvedValue({ data: { access_token: "t" } });
    await refreshAccessToken();
    await refreshAccessToken();
    expect(post).toHaveBeenCalledTimes(2);
  });
});

describe("error messages", () => {
  it("reads the API's error shape", () => {
    const error = { isAxiosError: true, response: { data: { error: { message: "Nope." } } } };
    expect(errorMessage(error)).toBe("Nope.");
  });

  it("explains a network failure", () => {
    expect(errorMessage({ isAxiosError: true })).toContain("Can't reach the server");
  });

  it("falls back for anything else", () => {
    expect(errorMessage(new Error("x"))).toBe("Something went wrong. Please try again.");
  });
});
