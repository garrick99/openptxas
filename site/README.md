# OpenPTXas technical showcase

Single-page React component presenting the current OpenPTXas milestone
(corpus green, SAXPY perf lead) as a clean technical site.

## What's here

- `OpenPTXasShowcase.jsx` — self-contained React component. Import and
  render it from any Tailwind-configured React app.

## Dependencies

Assumes the host project already has:

- `react` (>= 17, works with 18/19)
- `tailwindcss` (>= 3)
- `framer-motion` (>= 10)

No external CSS, no icon libraries — icons are inline SVG.

## Quickstart (Vite + Tailwind)

```bash
npm create vite@latest showcase -- --template react
cd showcase
npm install framer-motion tailwindcss @tailwindcss/vite
```

In `vite.config.js`, enable Tailwind. Then drop
`OpenPTXasShowcase.jsx` into `src/`, and in `src/App.jsx`:

```jsx
import OpenPTXasShowcase from "./OpenPTXasShowcase";
export default function App() {
  return <OpenPTXasShowcase />;
}
```

`npm run dev` and visit the shown URL.

## Quickstart (Next.js app router)

Drop the file into `app/components/OpenPTXasShowcase.jsx`. The component
uses `motion.section whileInView` which requires it to be a client
component — add `"use client"` at the top if using the Next.js app
router.

## Sections

| # | Section     | Purpose                                              |
|---|-------------|------------------------------------------------------|
| 1 | Hero        | Headline + 3 stat cards (142/142, bit-identical, 1.7×) |
| 2 | Stack       | Forge → OpenCUDA → PTX → OpenPTXas → SASS → GPU        |
| 3 | SAXPY       | Golden-child: code + comparison table                |
| 4 | Corpus      | Category grid + CI command                           |
| 5 | Timeline    | R31 → R58 with R48 keystone + R58 closing commit     |
| 6 | Principles  | Proof-first / narrow / GPU-verified / no-regressions |
| 7 | Next phase  | CI gate, perf parity, feature families               |

All content is inline — no data fetching, no external APIs. Edit the
`STACK`, `CORPUS_CATEGORIES`, `TIMELINE`, `NEXT_PHASE`, and
`HERO_STATS` arrays near the top of the component to update.
