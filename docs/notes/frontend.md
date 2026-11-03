# Frontend

Code: `frontend/`.

## How to run

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173, proxying /api to localhost:8000
```

## How to test

```bash
npm test            # Vitest and Testing Library: chat, upload, the API client
npm run typecheck
npm run lint
npm run build
```

Component tests stub the API. Only chat and upload have component tests, so the other pages are covered by type checks and a build.

## Key decisions

**1. The access token lives in memory, and refresh is single-flight.** A token in `localStorage` can be read by any injected script. In memory it disappears on reload, so the app silently refreshes from the `httpOnly` cookie on startup. Refresh tokens work once, and the backend treats reuse as theft, so parallel requests that all hit a 401 must share one refresh. `refreshAccessToken` keeps one promise for all callers, and `client.test.ts` proves it.

**2. Streaming uses `fetch`, not `EventSource`.** `EventSource` can only send GET and cannot set an `Authorization` header. `fetch` gives a readable stream, and `AbortController` gives the stop button: aborting closes the connection, and the server keeps what was written so far.

**3. Server state and client state are separate.** React Query owns anything from the API, with caching and optimistic updates that roll back on failure. Zustand holds only the session, theme and toasts. *Rejected:* putting API data in a global store, which means rebuilding caching and invalidation by hand.

## Where things are

| Feature | Where |
|---|---|
| Role-based routing | `RequireAuth` in `components/Shell.tsx`. The server still enforces every rule |
| Optimistic updates | Conversation rename and delete, plan items, reminders |
| Token refresh | The axios interceptor in `api/client.ts` |
| Dark mode | `[data-theme="dark"]` token set in `styles/app.css` |
| Down to 360px | Mobile-first layouts. Tables scroll inside their own container |
| Keyboard and ARIA | Native controls, a skip link, named landmarks, `role="log"` for chat, a progress bar with values |
| Skeletons and empty states | Every data region has loading, empty, error and full states |
| Error boundary | `ErrorBoundary` in `components/ui.tsx` |

## Design

Tokens are in `styles/tokens.css`, and Tailwind maps to them in `tailwind.config.js`, so no colour or font is written anywhere else. The one object the app is known by is the cited answer: a top-lit panel set in Figtree, with numbered sources as pills that open beside it. Headings use Instrument Sans. Light mode is a warm paper with one teal, and dark mode a blue-black with one amber, both defined in `styles/app.css`.

## Known limits

- Stop closes the connection, but the agent's current LLM call still finishes on the server.
- Search results open the conversation, not the exact message.
- The planner has one fixed day start, as the backend does.
