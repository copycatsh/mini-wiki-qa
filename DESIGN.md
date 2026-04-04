# Design System — Mini Wiki Q&A

## Product Context
- **What this is:** RAG-powered Q&A system that answers questions using local documents with semantic search, reranking, and LangGraph orchestration
- **Who it's for:** Developers and researchers querying local document collections
- **Space/industry:** AI/ML developer tools, document Q&A, RAG systems
- **Project type:** Web app (chat interface with settings sidebar)

## Aesthetic Direction
- **Direction:** Brutally Minimal with warmth
- **Decoration level:** Minimal — typography and whitespace do the work, no gradients or decorative elements
- **Mood:** A warm, approachable research tool. Claude.ai's calm confidence meets a well-designed library. Academic gravitas without being stuffy. The interface disappears so the conversation stays front and center.
- **Reference sites:** Claude.ai (warm minimalism, terra cotta accent, serif body), Perplexity (citations UX, paper-white background, clean layout)

## Typography
- **Display/Hero:** Instrument Serif — Academic, research-library gravitas. Used for app title and empty-state headlines only.
- **Body (AI responses):** Source Sans 3 — Excellent readability for long-form answers. OpenType features for proper numerals. Not overused in the AI tool space.
- **UI/Labels:** Geist — Clean, modern, designed for interfaces. Buttons, sidebar, navigation, form labels.
- **Data/Tables:** Geist (tabular-nums feature enabled)
- **Code/Metadata:** Geist Mono — Latency tags, chunk counts, pipeline labels, code snippets.
- **Loading:** Google Fonts CDN for Instrument Serif and Source Sans 3. Geist via npm (`geist` package, Next.js native support).
- **Scale:** 1.25 modular ratio, 14px base
  - 48px — Display (Instrument Serif, app title)
  - 32px — Display secondary (Instrument Serif, empty-state)
  - 22px — Heading (Source Sans 3, 600 weight)
  - 17px — Subheading (Source Sans 3, 500 weight)
  - 15px — Body (Source Sans 3, AI response text)
  - 14px — UI base (Geist, controls and labels)
  - 13px — Mono (Geist Mono, metadata and code)
  - 11px — Caption/Label (Geist, uppercase, 0.08em tracking)

## Color
- **Approach:** Restrained — 1 accent + warm neutrals. Color is rare and meaningful.

### Light Mode
- **Background:** #FAF9F6 — warm off-white, reduces eye strain for long reading sessions
- **Surface:** #FFFFFF — cards, citation blocks, elevated elements
- **Sidebar:** #F5F3EF — subtle distinction from main area
- **Code BG:** #F5F3EF — inline code and code block backgrounds
- **Border:** #E8E5E0 — warm gray, not cool
- **Text:** #1A1A1A — near-black, easier on eyes than pure #000
- **Text muted:** #6B6B6B — secondary text, timestamps, metadata labels
- **Accent:** #C2713A — warm amber/terra cotta, nods to Claude without copying
- **Accent hover:** #A85E30 — darker shade for interactive states

### Dark Mode
- **Background:** #1A1917 — warm dark, not blue-black
- **Surface:** #242320 — elevated elements
- **Code BG:** #2A2927
- **Border:** #3A3835
- **Text:** #E8E5E0 — warm off-white
- **Text muted:** #9A9890
- **Accent:** #D4845A — lightened for dark backgrounds, maintains warmth
- **Accent hover:** #E09468

### Semantic Colors
- **Success:** #2D8A4E (light bg: #f0faf4, dark bg: #0d2818)
- **Warning:** #B8860B (light bg: #fdf8e8, dark bg: #2d2200)
- **Error:** #C53030 (light bg: #fef2f2, dark bg: #2d0f0f)
- **Info:** #2B6CB0 (light bg: #eff6ff, dark bg: #0d1f3c)

## Spacing
- **Base unit:** 8px
- **Density:** Comfortable — not cramped like a dashboard, not spacious like a marketing site
- **Scale:** 2xs(2) xs(4) sm(8) md(16) lg(24) xl(32) 2xl(48) 3xl(64)
- **Chat message gap:** 16px vertical between messages
- **Sidebar padding:** 20px vertical, 16px horizontal
- **Chat area padding:** 24px vertical, 32px horizontal

## Layout
- **Approach:** Grid-disciplined
- **Sidebar:** 240px collapsible left panel, contains settings (pipeline selector, rerank toggle, model info)
- **Chat column:** Centered, max-width 720px
- **Input bar:** Pinned to bottom, same max-width as chat column
- **Border radius:** sm: 4px (buttons, tags, inputs), md: 8px (cards, chat input, send button), lg: 12px (mockup container, modal), full: 9999px (toggle switches, pills)
- **Responsive:** Sidebar collapses below 768px viewport width

## Motion
- **Approach:** Minimal-functional — only transitions that aid comprehension
- **Easing:** enter(ease-out) exit(ease-in) move(ease-in-out)
- **Duration:** micro(50-100ms) short(150-250ms) medium(250-400ms)
- **Specific animations:**
  - Sidebar collapse/expand: 200ms ease-out
  - Citation accordion expand: 150ms ease-out
  - Streaming cursor blink: CSS animation
  - Button/input focus: 150ms border-color transition
  - Theme toggle: CSS custom property transition on all colors
- **Not used:** No entrance animations, no scroll effects, no page transitions

## Information Architecture

### Visual Hierarchy
1. **Sidebar** (left, 240px): App title (loudest) → Pipeline selector (primary) → Rerank toggle (secondary) → Model/index info (tertiary). Leave vertical space above settings for future "Recent conversations" section.
2. **Chat area** (center, max 720px): AI response text (primary) → Citations (secondary, collapsed) → Metadata tags (tertiary, ambient)
3. **Input bar** (bottom, pinned): Input field + send button. Disabled during streaming with "Stop" button.

### Message Layout
- Full-width messages spanning the chat column, no bubbles or background tinting
- "You" and "Mini Wiki" labels in 11px uppercase Geist above each message
- Dual-font strategy provides visual distinction (Geist for user, Source Sans 3 for AI)

## Interaction States

### Streaming (requires SSE backend)
- User sends query → input disables, "Stop" button replaces "Send"
- "Mini Wiki is thinking..." with pulsing dot animation (accent color, 1s cycle)
- Tokens stream in via SSE, rendered in Source Sans 3
- After stream completes: citations accordion + metadata tags fade in (150ms)
- If user clicks "Stop": partial response shown, metadata shows "stopped"

### Empty State (first visit)
- Center of chat area: "Ask your documents anything" in Instrument Serif 32px
- Below: 3-4 suggestion chips (example queries from indexed documents)
- Below chips: muted text "365 chunks indexed from 12 documents" (dynamic from /health endpoint)
- Chips styled as secondary buttons (border, no fill, sm radius)

### Error States
- **Network/500:** Red alert inline in chat: "Something went wrong. Check that the backend is running." + "Retry" button
- **Injection guard blocked:** Warning alert: "This query was blocked by the safety filter. Try rephrasing." No retry.
- **No documents indexed:** Empty state changes to: "No documents indexed yet. Run the ingestion pipeline first." in muted text
- **Backend unreachable on load:** Full-page centered message with error alert styling

### Loading
- Initial page load: skeleton for sidebar (3 lines), chat area shows empty state immediately
- Query in progress: thinking indicator + disabled input (see Streaming above)

### Citations (expanded)
- Each citation shows: document name (Geist 12px, accent color), chunk text preview (Source Sans 3 13px, muted, first 200 chars, expandable), relevance score (Geist Mono 11px, as percentage)
- Score visualization: subtle background bar proportional to score, using 8% accent tint
- "Show full text" link for chunks longer than 200 chars

## Responsive Design

### Breakpoints
- **Desktop (≥1024px):** Full layout with persistent sidebar (240px)
- **Tablet (768-1023px):** Sidebar collapses to icon rail (48px) with tooltips on hover
- **Mobile (<768px):** Sidebar hidden. Gear icon in top-right header opens bottom sheet with pipeline selector + rerank toggle. Chat area is full-width with 16px horizontal padding.

### Mobile-specific
- Input bar: full-width with 12px padding, send button stays 40x40
- Message labels stack vertically
- Citations accordion: chunk text preview truncated to 120 chars
- Metadata tags wrap to second line if needed
- Bottom sheet: 280px max height, drag handle at top, backdrop blur

## Accessibility

### Keyboard Navigation
- Tab order: sidebar controls → chat message list → input → send button
- Enter in input field sends query (Shift+Enter for newline if multiline supported)
- Escape closes bottom sheet (mobile) or collapses expanded citations
- Arrow keys navigate between messages in chat (optional, not required for MVP)

### Screen Readers
- Chat container: `role="log"`, `aria-live="polite"` for new messages
- Sidebar controls: explicit `aria-label` on toggle ("Enable reranking") and select ("Select pipeline")
- Citations toggle: `aria-expanded` state, `aria-controls` pointing to citation list
- Streaming: `aria-busy="true"` on chat container during streaming
- Send button: `aria-label="Send message"`

### Color Contrast
- Primary text #1A1A1A on #FAF9F6: 15.2:1 (AAA pass)
- Muted text #6B6B6B on #FAF9F6: 4.7:1 (AA pass)
- Accent #C2713A: use only for interactive elements, icons, and large text. Do NOT use as small body text color (3.8:1 on white, borderline AA)
- All focus rings: 2px solid accent, 2px offset

### Touch Targets
- Minimum 44x44px for all interactive elements
- Send button: 40x40px (increase to 44x44 on mobile)
- Toggle switch: 40x22px tappable area, but hit target extends to 44px height via padding

## Component Patterns

### Chat Messages
- User messages: Geist font, 14px, 500 weight, full-width, "You" label above
- AI messages: Source Sans 3, 15px, 400 weight, 1.7 line-height, full-width, "Mini Wiki" label above
- Message labels: 11px uppercase Geist, muted color, 6px bottom margin
- Code inline: Geist Mono, 13px, code-bg background, 3px radius

### Citations Accordion
- Toggle button: code-bg background, 12px Geist, muted text
- Arrow rotates 90deg on expand (150ms ease-out)
- Citation list: surface background, 1px border, sm radius
- Citation numbers: Geist Mono, accent color, code-bg background

### Metadata Tags
- Geist Mono, 11px, code-bg background, sm radius, 2px/8px padding
- Accent variant (pipeline label only): 12% accent color background, accent text
- Semantic variant: 12% semantic color background
- Plain tags for latency and chunk count (no accent, just muted on code-bg)

### Alerts
- 3px left border, semantic color
- Light tinted background matching semantic color
- Source Sans 3, 14px body text

### Form Controls
- Input: 1px border, sm radius, accent border on focus
- Select: same as input, no native appearance
- Toggle: 40x22px, accent background when active, 200ms transition
- Send button: 40x40px, accent background, md radius, white arrow icon

## Implementation Notes

### shadcn/ui Font Override
The shadcn/ui default font is Inter. Override in `tailwind.config.ts`:
- `fontFamily.sans` → Geist (UI font)
- Add custom `fontFamily.serif` → Instrument Serif (display)
- Add custom `fontFamily.body` → Source Sans 3 (AI responses)
- Add custom `fontFamily.mono` → Geist Mono (metadata/code)

### Streaming Backend
SSE streaming is implemented via `/ask/stream` and `/ask-graph/stream` endpoints returning `text/event-stream`. Frontend consumes via `fetch` with `ReadableStream` in the `useChat` hook. Events: `token`, `citations`, `metadata`, `error`, `done`.

### Conversation Persistence (deferred)
MVP has no persistence. Page refresh clears chat. Sidebar is designed to accommodate a future "Recent conversations" list above settings. No localStorage, no backend storage for MVP.

## NOT in Scope (explicitly deferred)
- Conversation history / persistence (sidebar space reserved for later)
- Multi-model selection (only one LLM backend at a time)
- Document upload from the frontend (ingestion is CLI/API only)
- User authentication (single-user tool)
- Dark mode auto-detection via `prefers-color-scheme` (manual toggle only for MVP)

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-04 | Initial design system created | Created by /design-consultation based on Claude.ai inspiration + competitive research of Perplexity, ChatGPT, Phind |
| 2026-04-04 | Instrument Serif for display | No AI chat uses serif for display. Gives academic "research library" feel that fits document Q&A. |
| 2026-04-04 | Warm amber accent #C2713A | Avoids blue/purple developer tool default. Says "warm research tool" not "cold dev utility." |
| 2026-04-04 | Dual-font body strategy | Source Sans 3 for AI responses (reading-optimized), Geist for UI chrome (interface-optimized). Subtle hierarchy. |
| 2026-04-04 | Off-white #FAF9F6 background | Matches user request, close to Claude's Pampas #F4F3EE. Proven to reduce eye strain. |
| 2026-04-04 | Full-width messages with labels | No bubbles or background tinting. Dual-font strategy provides visual distinction. Matches Claude.ai pattern. |
| 2026-04-04 | Real SSE streaming over fake | Trust is earned at the pixel level. Don't fake streaming if backend is synchronous. Build real streaming. |
| 2026-04-04 | Hero headline + suggestion chips for empty state | Empty states are features. Instrument Serif headline, query chips, and index stats give first-time users a fast on-ramp. |
| 2026-04-04 | Inline chunk text in citations | Citations show actual chunk text (200 char preview) + relevance score. Trust mechanism for RAG answers. |
| 2026-04-04 | Header icon → bottom sheet on mobile | Gear icon replaces sidebar on mobile. Bottom sheet for 2 controls is lighter than hamburger menu. |
| 2026-04-04 | No conversation persistence for MVP | Page refresh clears chat. Sidebar reserves space for future conversation list. Subtraction default. |
