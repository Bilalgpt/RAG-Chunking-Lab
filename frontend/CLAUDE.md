# Frontend — React + Tailwind

## What Lives Here
Vite + React 18 single-page application. All UI for technique selection, document viewing, chunk visualization, chat, comparison mode, and metrics.

## Directory Layout
- `src/components/` — all React components (see components/CLAUDE.md)
- `src/hooks/` — custom hooks, including `useLLMConfig` and `useChunking`
- `src/pages/` — top-level page components (currently one page: Lab)
- `src/utils/` — pure utility functions (formatting, color mapping, etc.)
- `src/api.js` — all fetch calls to the backend, one function per endpoint

## API Key State — Critical Rule
The user's LLM provider choice and API key live in a single React state object managed by `useLLMConfig` hook. They are:
- Never written to `localStorage`, `sessionStorage`, or any cookie.
- Never sent to any endpoint except as request headers (`X-API-Key`, `X-LLM-Provider`).
- Cleared on page refresh by design.

## Styling Rules
- Use Tailwind utility classes only — no custom CSS files except `index.css` (Tailwind directives).
- Dark mode is class-based (`dark:` prefix). Default to dark theme.
- Color coding for chunking techniques: each technique gets a consistent accent color defined in `src/utils/techniqueColors.js`.

## State Management
- No Redux or Zustand. React `useState` + `useContext` is sufficient for this scope.
- A single `LabContext` provides: selected technique, document, LLM config, and current chunks.

## API Communication
- All backend calls go through `src/api.js`. Never fetch directly inside components.
- Backend base URL defaults to `http://localhost:8000`; overridable via `VITE_API_BASE_URL` env var.
