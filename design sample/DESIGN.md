# Design System Strategy: The Quantitative Precision Editorial

This design system is engineered for a sophisticated demographic: the Japanese horse racing analyst and quantitative investor. It moves away from the chaotic, high-stimulus aesthetic of traditional gambling sites, instead adopting the "Digital Curator" persona—an authoritative, data-driven editorial experience that feels more like a Bloomberg terminal or a high-end financial journal.

## 1. Creative North Star: "The Digital Curator"
The system is built on the philosophy of **"Quiet Authority."** In a field crowded with noise, we win through clarity, intentional asymmetry, and depth. We break the "template" look by using generous white space (breathing room) contrasted against high-density data modules. 

Key design pillars include:
- **Intentional Layering:** Depth is created through surface shifts, not lines.
- **Editorial Typography:** Large, high-contrast numerical displays that treat data as art.
- **Micro-density:** Maintaining the "information density" Japanese users expect, but organizing it into clean, logical nested containers.

## 2. Color & Surface Philosophy

### The "No-Line" Rule
To achieve a premium feel, **1px solid borders are strictly prohibited for sectioning.** Boundaries must be defined solely through background color shifts.
*   *Implementation:* Place a `surface_container_lowest` card on a `surface_container_low` background. The contrast in tone creates the edge, making the UI feel integrated and sophisticated rather than "boxy."

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers—like stacked sheets of fine paper.
- **Base Layer:** `surface` (#f9f9fb)
- **Content Sections:** `surface_container_low` (#f3f3f5)
- **High-Priority Cards:** `surface_container_lowest` (#ffffff)
- **Interactive Elements:** `surface_container_high` (#e8e8ea)

### The Glass & Gradient Rule
For main CTAs and performance headers, use subtle gradients transitioning from `primary` (#000666) to `primary_container` (#1a237e). For floating navigation or modal overlays, use **Glassmorphism**: a semi-transparent `surface` color with a `backdrop-blur` of 12px–20px to allow the underlying data colors to bleed through softly.

## 3. Typography: The Data Narrative

Our typography pairs the technical precision of **Manrope/WorkSans** for numbers with the reliable readability of **Inter/Noto Sans JP** for Japanese text.

*   **Display (Manrope):** Use `display-lg` and `display-md` for "Return on Investment" (ROI) or "Strike Rate" percentages. These should be the "hero" of the page.
*   **Headlines (Manrope):** `headline-sm` should be used for section titles to provide a modern, data-driven look.
*   **Body (Inter):** `body-md` is our workhorse for analysis text. Ensure line-height is set to 1.6x for maximum readability.
*   **Labels (WorkSans):** Use `label-md` in All Caps (for Latin characters) or bold weight for status badges (e.g., "WIN", "PLACE").

**Visual Emphasis:** Numbers (payouts, odds) should always be 1-2 steps larger in the hierarchy than their accompanying labels to signify their importance to the investor.

## 4. Elevation & Depth

### The Layering Principle
Hierarchy is achieved through "Tonal Layering."
- **Level 0:** `surface`
- **Level 1:** `surface_container_low` (General sections)
- **Level 2:** `surface_container_lowest` (Individual cards)

### Ambient Shadows
When a card must "float" (e.g., a hovered prediction card), use an **Ambient Shadow**:
- `Color: on_surface` (at 6% opacity)
- `Blur: 24px`
- `Offset: Y=8px`
This mimics natural light and avoids the "cheap" look of heavy black shadows.

### The "Ghost Border" Fallback
If a divider is required for accessibility, use a **Ghost Border**: `outline_variant` (#c6c5d4) at **15% opacity**. Never use a 100% opaque border.

## 5. Components

### Cards & Data Modules
*   **Layout:** Use `surface_container_lowest` as the card background.
*   **Spacing:** Use `spacing-5` (1.1rem) for internal padding.
*   **No Dividers:** Separate card header from body using a slight background shift (`surface_container_low` for the header) instead of a line.

### Status Badges (The "Performance" Chips)
*   **Success (Win):** `secondary_container` (#a0f399) background with `on_secondary_container` (#217128) text.
*   **High Yield:** `tertiary_container` (#cba72f) with `on_tertiary_container` (#4e3d00) text.
*   **Shape:** Use `rounded-sm` (0.125rem) for a professional, "ticket-like" feel rather than fully rounded pills.

### Buttons
*   **Primary:** Background gradient (`primary` to `primary_container`). Text: `on_primary`. Shape: `rounded-md` (0.375rem).
*   **Secondary:** Background: `surface_container_high`. Text: `primary`. 
*   **Tertiary:** No background. Text: `primary`. Use for low-emphasis actions like "View More Details."

### Performance Graphs
*   **Line Color:** Use `secondary` (#1b6d24) for growth lines.
*   **Fill:** Use a gradient from `secondary` (20% opacity) to transparent.
*   **Grid Lines:** Use `outline_variant` at 10% opacity.

## 6. Do's and Don'ts

### Do
*   **DO** use `tertiary_fixed` (Gold) sparingly. It is for "Premium" or "High-Return" highlights only. Overuse diminishes its perceived value.
*   **DO** maintain high information density in data tables, but use the `spacing-px` and `spacing-0.5` scale to create clear horizontal alignment.
*   **DO** use `surface_bright` to highlight the most important analytical takeaway on a page.

### Don't
*   **DON'T** use pure black (#000000) for text. Use `on_surface` (#1a1c1d) to maintain a soft, premium editorial feel.
*   **DON'T** use traditional "Drop Shadows" on cards. Stick to tonal shifts or Ambient Shadows.
*   **DON'T** use red for anything other than "Error" or "Loss." In the Japanese market, red can be confusing in a racing context; stick to `secondary` (Green) for positive outcomes and `primary` (Navy) for neutral/stable data.