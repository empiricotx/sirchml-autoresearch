# Frontend Design Transfer Spec

## 1. Overall Design Intent
- Build for a professional internal scientific platform, not a consumer marketing site.
- Visual style is clean, dense, and functional.
- Prioritize clarity, scanability, and fast task completion over decoration.
- Use restrained motion and subtle depth, not flashy animation.
- Interfaces should feel trustworthy, structured, and slightly technical.

## 2. Core Visual Language
- Primary palette is deep blue with cool gray neutrals.
- Backgrounds are mostly white or very light gray.
- Accent color usage is sparse and purposeful.
- Cards, tables, and panels are the main organizing surfaces.
- Borders are used heavily to define structure; shadows are light and secondary.
- Rounded corners are present but modest, not soft or playful.

## 3. Tokens
- Primary brand: rgb(0, 51, 78)
- Secondary brand: rgb(20, 83, 116)
- Tertiary brand: rgb(85, 136, 163)
- Light gray surface: rgb(232, 232, 232)
- Medium gray divider: rgb(204, 204, 204)
- Default radius: 8px
- Typography: sans-serif for UI, monospace for IDs / technical values

## 4. Layout Rules
- Use a fixed or sticky dark header as the main app frame anchor.
- Keep content inside a centered container with generous horizontal padding.
- Prefer card grids, dashboard sections, and bordered panels.
- Use whitespace to separate sections rather than heavy ornament.
- Tables should support sticky headers and scroll within bounded containers.

## 5. Component Patterns
- Buttons: solid primary for main actions, outline/ghost for secondary actions.
- Cards: white background, light border, subtle shadow, slight hover lift when clickable.
- Search inputs: rounded, prominent, icon-led, primary-colored focus treatment.
- Status indicators: compact pills/badges with muted background fills.
- Data blocks: use monospace for IDs, sequence-like values, or machine-readable fields.

## 6. Interaction Style
- Hover states are subtle: small shadow increase, slight scale/lift, color darkening.
- Focus states should use the primary blue ring.
- Loading states should use skeletons or simple spinners.
- Motion should be fast and minimal.

## 7. What To Avoid
- No loud gradients, neon accents, or highly saturated UI.
- No oversized hero-style marketing sections.
- No glassmorphism-heavy or overly soft consumer styling.
- No inconsistent border radius or ad hoc color choices.
- Do not invent new visual patterns when an existing card/table/filter pattern works.

## 8. Implementation Bias
- Prefer Tailwind utility composition with shared primitives.
- Use CSS variables for theme tokens.
- Reuse card/button/input primitives across pages.
- Preserve a shadcn-style base component system.
