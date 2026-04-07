# Components — Conventions

## Component Inventory (planned)
- `TechniqueSelector` — grid of 7 cards, one per chunking technique
- `DocumentViewer` — displays document text with chunk boundaries highlighted
- `ChunkPanel` — list of retrieved chunks with relevance scores
- `ChatInterface` — message thread + input, sends query to backend
- `MetricsPanel` — shows chunk count, avg chunk size, retrieval latency
- `ComparisonView` — side-by-side two-technique layout
- `APIKeyInput` — provider selector + key input, fires `useLLMConfig` setter
- `ChunkBadge` — small colored label showing chunk index and technique

## Rules
- One component per file. File name matches component name (PascalCase).
- Props are destructured at the top of the function, not inline.
- No component fetches data directly — use hooks from `src/hooks/` or callbacks passed as props.
- No component reads from or writes to `localStorage` — API key rule (see `frontend/CLAUDE.md`).
- Keep components under ~150 lines. Extract sub-components if growing beyond that.

## Chunk Highlighting
`DocumentViewer` receives an array of `{start, end, chunkIndex}` span objects and uses inline `<mark>` elements with technique-specific colors from `src/utils/techniqueColors.js`.
