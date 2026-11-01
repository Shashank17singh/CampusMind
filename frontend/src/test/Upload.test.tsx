import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { UploadPanel } from "../pages/Faculty";

function renderPanel() {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <UploadPanel />
    </QueryClientProvider>,
  );
}

const pdf = (name = "circular.pdf") => new File(["data"], name, { type: "application/pdf" });

const uploaded = (fields: object) => ({
  id: "d1",
  status: "uploaded",
  progress: 0,
  error_message: null,
  duplicate_of: null,
  ...fields,
});

describe("upload", () => {
  beforeEach(() => {
    vi.spyOn(api, "post").mockResolvedValue({ data: uploaded({}) });
  });
  afterEach(() => vi.useRealTimers());

  it("uploads a chosen file and shows a progress bar", async () => {
    const user = userEvent.setup();
    renderPanel();
    await user.upload(screen.getByLabelText("Choose files to upload"), pdf());
    expect(api.post).toHaveBeenCalledWith("/documents", expect.any(FormData), expect.anything());
    expect(
      await screen.findByRole("progressbar", { name: "circular.pdf progress" }),
    ).toBeInTheDocument();
  });

  it("takes several files dropped at once", async () => {
    renderPanel();
    fireEvent.drop(screen.getByTestId("dropzone"), {
      dataTransfer: { files: [pdf("a.pdf"), pdf("b.pdf")] },
    });
    expect(await screen.findByRole("progressbar", { name: "a.pdf progress" })).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "b.pdf progress" })).toBeInTheDocument();
    expect(api.post).toHaveBeenCalledTimes(2);
  });

  it("reports upload progress on the bar", async () => {
    vi.mocked(api.post).mockImplementation((_url, _data, config) => {
      config?.onUploadProgress?.({ loaded: 2, total: 4 } as never);
      return new Promise(() => undefined);
    });
    renderPanel();
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [pdf()] } });
    const bar = await screen.findByRole("progressbar");
    await waitFor(() => expect(bar).toHaveAttribute("aria-valuenow", "50"));
  });

  it("follows ingestion status until the document is ready", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const get = vi
      .spyOn(api, "get")
      .mockResolvedValueOnce({ data: { status: "embedding", progress: 70, error_message: null } })
      .mockResolvedValueOnce({ data: { status: "ready", progress: 100, error_message: null } });
    renderPanel();
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [pdf()] } });
    await screen.findByText(/Queued/);

    await act(() => vi.advanceTimersByTimeAsync(2100));
    await screen.findByText(/Indexing/);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "70");

    await act(() => vi.advanceTimersByTimeAsync(2100));
    await screen.findByText(/Ready/);

    await act(() => vi.advanceTimersByTimeAsync(5000));
    expect(get).toHaveBeenCalledTimes(2);
  });

  it("shows the reason when ingestion fails", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(api, "get").mockResolvedValue({
      data: { status: "failed", progress: 0, error_message: "This PDF is corrupted." },
    });
    renderPanel();
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [pdf()] } });
    await screen.findByText(/Queued/);
    await act(() => vi.advanceTimersByTimeAsync(2100));
    expect(await screen.findByText("This PDF is corrupted.")).toBeInTheDocument();
  });

  it("says when the file is already uploaded", async () => {
    vi.mocked(api.post).mockResolvedValue({
      data: uploaded({ status: "ready", progress: 100, duplicate_of: "d0" }),
    });
    renderPanel();
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [pdf()] } });
    expect(await screen.findByText("This file is already uploaded.")).toBeInTheDocument();
  });

  it("shows the server's reason for a rejected file", async () => {
    vi.mocked(api.post).mockRejectedValue({
      isAxiosError: true,
      response: { data: { error: { message: "Unsupported file type." } } },
    });
    renderPanel();
    fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files: [pdf("a.exe")] } });
    expect(await screen.findByText("Unsupported file type.")).toBeInTheDocument();
    expect(screen.getByText(/Failed/)).toBeInTheDocument();
  });

  it("highlights the drop area while dragging", () => {
    renderPanel();
    const zone = screen.getByTestId("dropzone");
    fireEvent.dragOver(zone);
    expect(zone.className).toContain("border-accent");
    fireEvent.dragLeave(zone);
    expect(zone.className).not.toContain("border-accent");
  });
});
