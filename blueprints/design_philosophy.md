# Design System Inspired by Mastercard (White–Blue Edition)

## 1. Visual Theme & Atmosphere

This design system adapts Mastercard's editorial language into a crisp, institutional white-and-blue palette. The canvas is pure optical white (`#FFFFFF`) — clean, confident, clinical in the best sense. On top of that canvas, everything that matters is shaped like a stadium, a pill, or a perfect circle. The dominant visual gesture is the **oversized radius**: heroes carry 40-point corners, cards go fully pill-shaped, service images are cropped into circular orbits, and buttons either complete the pill or fit snugly at 20 points. There are almost no sharp corners anywhere on the page.

The second gesture is **orbit and trajectory**. Circular image masks don't sit still — they're connected by thin, hand-drawn-feeling blue arcs that span entire viewport widths, implying a constellation of services rather than a list. Each circle has a small attached "satellite" — a white micro-CTA holding an arrow icon — docked onto its perimeter like a moon. This is the most distinctive thing about this design language: the circles feel like they're in motion even though the page is still.

Typography is rendered entirely in **MarkForMC** (or its open-source substitute). Headlines are set at a medium weight (500) with tight negative letter-spacing (-2%), giving them confidence without shouting. Body copy runs at the same family in a slightly lighter weight (450) — a weight you rarely see on the web, chosen because it reads softer than regular 400 without feeling thin. The whole system — white surfaces, pill shapes, circular portraits, traced-blue orbits, navy CTAs — feels simultaneously institutional (a trusted global brand) and editorial (a modern brand magazine), which is exactly the tension this palette wants to hold.

**Key Characteristics:**
- Pure white canvas (`#FFFFFF`) as the default body background — bright, open, and airy
- Extreme border-radius as design language: 40px, 99px, 1000px dominate; anything square feels out of place
- Circular image portraits with attached white satellite-CTAs and traced-blue orbital paths
- Ghost "watermark" headlines (light-blue-on-white text at heading scale) layered behind circle portraits
- Navy primary CTAs with 20px radius in the body — the consent orange/blue is kept to compliance flows
- Floating pill-shaped navigation that docks below the viewport top with rounded shoulders
- Eyebrow labels with a tiny accent dot + uppercase bold tracking — used as the section-category signal
- Dark navy footer (`#0A1628`) with four-column link layout and large conversational headline

## 2. Color Palette & Roles

### Primary
- **Ocean Navy** (`#0A1628`): The deep warm-navy used for primary CTAs, headline text on white, and the footer surface. Dark enough to read as near-black but distinctly blue — never feels jet-black on the white canvas.
- **Brand Blue** (`#003087`): The primary brand identity color. Used in the logo and as a strong accent for key UI moments — section dividers, active states, and selected indicators.

### Secondary & Accent
- **Signal Blue** (`#0070D2`): The vivid action blue used on primary interactive elements and eyebrow dots. Bright, trustworthy, and attention-directing — the page's primary energetic color.
- **Sky Blue** (`#5BA4E0`): A lighter, airier blue used for carousel active indicators and decorative orbital arcs. Always acts as an attention cue, never as body color.
- **Deep Indigo** (`#1A3A6B`): The deep navy-blue used for secondary link-style buttons. Sits between ocean navy and signal blue.

### Surface & Background
- **Canvas White** (`#FFFFFF`): The page canvas. Bright, clean, the default body background. All editorial sections sit on this.
- **Lifted White** (`#F7F9FC`): One step off-white — used for nested "raised" sections that want to feel like paper laid on paper. Carries a faint blue tint.
- **Ice White** (`#EBF2FA`): A light blue-tinted surface used inside a handful of component subregions and card backgrounds.
- **Soft Blue** (`#E8F1FB`): A very pale blue used for subtle hover states and background washes on feature sections.

### Neutrals & Text
- **Ocean Navy** (`#0A1628`): Primary headline and body text color.
- **Charcoal Blue** (`#1E2D45`): A slightly softer dark blue-black used for some text alternates.
- **Slate** (`#536378`): Muted secondary text — eyebrow label alternative, disabled states, small-print.
- **Steel** (`#6B7D92`) and **Pewter** (`#7A8FA6`): Deeper gray-blue for inline body accents and link alternates.
- **Mist** (`#B8C8D8`): Very muted blue-gray used for disabled or "whisper" text. Low contrast on white; use only for subdued content.

### Semantic & Accent
- **Link Blue** (`#0070D2`): The signal blue doubled as the inline link color. Saturated enough to read as a link without being neon.
- **Consent Teal** (`#006E7A`): A teal-adjacent action color used exclusively on consent flows and legal confirmations. Distinguishable from Signal Blue so it reads as a compliance-specific color.

### Gradient System
This system uses no programmatic gradients in the core UI. The visual impression of "depth" comes from two places:
- **Circular image portraits** where a cool-blue or white-toned subject fades to the white canvas at its edge
- **Deep card shadows** on elevated content (`rgba(10,22,40,0.08) 0px 24px 48px`) that create a soft halo beneath pill-shaped media

## 3. Typography Rules

### Font Family
- **Primary**: `MarkForMC` — geometric sans. Every headline, body paragraph, button, nav link, and footer link on the page.
- **Secondary**: `MarkOffcForMC` — an "Official" cut used in a minority of contexts (legal text, some forms).
- **Fallback stack**: `SofiaSans, Arial, sans-serif`

### Hierarchy

| Role | Size | Weight | Line Height | Letter Spacing | Notes |
|------|------|--------|-------------|----------------|-------|
| H1 (hero) | 64px | 500 | 64px | -1.28px (-2%) | Set to `1:1` line-height for very tight vertical rhythm on multi-line hero |
| H2 (section) | 36px | 500 | 44px | -0.72px (-2%) | Used in ghost-watermark headline treatments and section titles |
| H3 (card title) | 24px | 500 | 28.8px (1.2) | -0.48px (-2%) | Titles inside service/solution cards |
| H4 (subhead) | 14px | 700 | 18.2px (1.3) | normal | Rarely used in marketing surfaces |
| Eyebrow (H5) | 14px | 700 | 14px | 0.56px (+4%) | Uppercase, paired with a tiny accent dot (e.g., "• SERVICES") |
| Body paragraph | 16px | 450 | 22.4px (1.4) | normal | The half-step 450 weight is the signature — softer than 500, firmer than 400 |
| Nav link / Button label | 16px | 500 | 16px | -0.48px (-3%) | Tight, compact, no text-transform |
| Footer link | 14px | 450 | ~20px | normal | Lighter weight on dark footer for airier density |
| Footer column header | 12–14px | 700 | 14px | 0.56px (+4%) | Uppercase, muted gray-blue, short tracking |

### Principles
- **Weight 450 is load-bearing**. Most brands use 400/500/700; this system uses 450 for body copy, which creates an unusually soft reading tone.
- **Tight negative tracking on headlines** (-2%) gives display text its editorial density.
- **Uppercase tracking only on the eyebrow scale** (14px / 700 / +4% tracking). Don't use uppercase anywhere else.
- **One-font system**. The contrast comes from scale, weight, and letter-spacing, not from a serif or display accent.
- **Line-height ratio drops with size**. H1 is 1:1, H3 is 1.2, body is 1.4.

### Note on Font Substitutes
MarkForMC is proprietary and licensed. When rebuilding a matching aesthetic without access to the original:
- **Sofia Sans** (Google Fonts) is the closest open-source match.
- **Inter** at weights 450/500/700 works as a generic stand-in.
- **Neue Haas Grotesk** or **Geist** can approximate the geometric feel for commercial projects.
- Whichever substitute is used, preserve the **-2% letter-spacing on headlines** and the **450 body weight**.

## 4. Component Stylings

### Buttons

**Primary — Navy Pill**
- Background: Ocean Navy (`#0A1628`)
- Text: White (`#FFFFFF`)
- Border: 1.5px solid Ocean Navy (same as bg, creates crisp edge)
- Radius: 20px
- Padding: 6px 24px
- Font: MarkForMC 16px / weight 500 / letter-spacing -0.32px
- Default: solid navy pill on white canvas
- Active / pressed: subtle inward-shrink or 2px offset
- Use for: all marketing CTAs in the page body ("Learn more", "Explore", "Discover")

**Secondary — Outlined Pill**
- Background: White (`#FFFFFF`)
- Text: Ocean Navy (`#0A1628`)
- Border: 1.5px solid Ocean Navy
- Radius: 20px
- Padding: 6px 24px
- Font: MarkForMC 16px / weight 450 / line-height 20.8px
- Default: white-on-white pill with crisp navy outline
- Use for: secondary actions paired with a primary, or standalone utility CTAs

**Consent / Signal — Teal Pill**
- Background: Consent Teal (`#006E7A`)
- Text: White (`#FFFFFF`)
- Border: 0
- Radius: 24px
- Padding: 1px 30px
- Font: MarkForMC 13px / weight 400 / letter-spacing 0.13px
- Default: deep teal pill with white text
- Use for: cookie consent, privacy preference, and other legally-distinct confirmations. **Do not** use this teal for marketing CTAs.

**Satellite — Circular Micro-CTA**
- Background: White (`#FFFFFF`)
- Icon: Ocean Navy arrow (`→`) at ~20px
- Border: none
- Radius: 50% (perfect circle)
- Size: ~50–60px diameter
- Shadow: none or very subtle
- Default: docks onto the bottom-right edge of a circular portrait, protruding partway outside the portrait's circle
- Use for: the primary entry point into service/solution cards; always paired with a circular portrait

**Icon-Only Circle Button (carousel, play/pause)**
- Background: transparent or white
- Icon: 10–20px centered
- Border: 1px solid Ocean Navy (when on white) or none (when over media)
- Radius: 50%
- Size: 40px diameter minimum for carousel controls; 80px for hero video play
- Use for: carousel pagination/play-pause, hero video play, search toggle

### Cards & Containers

**Hero Media Frame (Stadium)**
- Background: Dark video or full-bleed imagery (typically deep navy `#0A1628` or `#0D1F3C` behind video)
- Radius: 40px all corners
- Width: ~full viewport minus ~48px gutter on each side
- Height: ~60–70% of viewport
- Shadow: none (sits directly on canvas)
- Corners: the extreme 40px radius on a media element is the most iconic gesture — do not round less

**Service / Solution Portrait Card**
- Shape: Perfect circle (radius 50%) or ellipse (radius 999px / 1000px)
- Diameter: 260–340px desktop; ~220px mobile
- Image crop: square source, cropped to circle
- Attached element: White satellite circular CTA docked bottom-right, ~40% outside the portrait
- Eyebrow below: accent dot + uppercase label (e.g., "• SERVICES", "• SOLUTIONS") — dot in Sky Blue (`#5BA4E0`)
- Title below: H3 (24px / weight 500 / -2% tracking), 1–2 lines max
- Decorative orbit: thin ~1px Sky Blue curved line spanning from this card outward to the next

**Pill Carousel Card**
- Radius: 1000px (full pill) or 40px corners (rounded stadium)
- Width: ~40–60% of viewport
- Height: ~380–420px (portrait-pill orientation)
- Content: full-bleed photography with small overlaid chip labels
- Chip inside: White pill (~ 999px radius), Ocean Navy text, padding 8px 20px
- Large inline CTA inside: Navy Pill button, oversized (padding 16px 40px, radius 40px)

**Ghost Watermark Text Block**
- Font: MarkForMC 72–128px / weight 500 / tight -2% tracking
- Color: Ice White (`#EBF2FA`) — blue-tinted white-on-white
- Position: layered behind portrait circles, bleeding off the viewport edge
- Purpose: sets section theme without competing with foreground copy

### Inputs & Forms
The search input in the nav header:
- Initial state: a 48px circular button with a magnifier icon
- Expanded state: horizontal input field, border `1px solid` Ocean Navy at ~50% opacity, radius 999px, padding 12px 24px, white background

**Country/language selector (footer)**
- Background: Ocean Navy (same as footer)
- Text: White
- Border: 1px solid `rgba(255,255,255,0.4)`
- Radius: 999px (full pill)
- Icon: downward chevron on the right

### Navigation

**Floating Nav Pill (desktop)**
- Container: white-to-translucent-white pill floating below the very top of the viewport with a ~24px top margin
- Radius: 999px / 1000px (full pill)
- Padding: ~16px 40px internal
- Shadow: very soft (`rgba(10, 22, 40, 0.06) 0px 4px 24px 0px`) — just enough to lift it off the white canvas
- Content: brand logo left, primary link group center, search icon right
- Link spacing: ~48–56px gap between primary links
- Link style: Ocean Navy, weight 500, 16px, no underline, no pill surround until active

**Mobile Nav**
- The same pill shape but collapsed to: logo + hamburger menu button + search icon only
- Menu opens into a full-screen overlay with the primary links stacked vertically

### Image Treatment

- **Aspect ratios used**: 1:1 (all service portraits — cropped to circle), ~3:4 or ~4:5 (carousel pill cards), 16:9 or wider (hero video frame)
- **Full-bleed vs padded**: Hero is viewport-wide with gutters; service portraits are always centered in their column with generous whitespace around
- **Masking**: Aggressive circular masking is the defining treatment — square source images are cropped to perfect circles. Never use rectangular service imagery.
- **Lazy loading**: Standard `loading="lazy"` with a soft blur-up transition from a white placeholder

### Decorative Orbital Lines

A signature motif: thin (~1–1.5px) single-weight curved lines in Sky Blue (`#5BA4E0`) tracing arcs between circular portraits. These lines:
- Imply connection between service cards without literal arrows
- Span widths from ~200px up to full-viewport arcs
- Feel hand-drawn (subtle irregularity) rather than perfect CSS curves
- Appear only in sections with circular portrait content — never on pill sections, never in the footer

### Footer

- Background: Ocean Navy (`#0A1628`)
- Text: White
- Padding: 48px horizontal 100px / bottom 148px (very tall bottom space)
- Structure: large conversational H2 left-aligned, then a 4-column link grid below
- Column headers: uppercase, muted, weight 700, letter-spacing +4%
- Link rows: white, weight 450, 14px
- External link marker: a small upper-right arrow (`↗`) after link text
- Bottom row (below a 1px white-at-opacity divider): copyright + privacy small-print + country-language pill dropdown + four social icons

## 5. Layout Principles

### Spacing System
- **Base unit**: 8px
- **Scale**: 8 / 16 / 24 / 32 / 48 / 64 / 96 / 128
- **Section vertical padding**: ~96–128px between major sections on desktop; ~48–64px on mobile
- **Card internal padding**: 32–40px on desktop, ~24px on mobile
- **Nav top margin**: ~24px from viewport top (the pill floats, doesn't touch)

### Grid & Container
- **Max content width**: ~1200–1280px centered, with ~48–100px horizontal gutter
- **Column pattern**: 12-column implied; practical layouts use 2-up asymmetric (large headline left, supporting text right), 1-up full-bleed (hero, video), or staggered single-portrait placement
- **Footer grid**: 4 equal columns on desktop, collapses to single column accordion on mobile

### Whitespace Philosophy
Whitespace is structure, not absence. A typical service section has:
- A ghost headline occupying the top ~40% of the section (mostly white)
- A single circular portrait positioned ~60% down, asymmetric to left or right
- ~300–500px of blank canvas between the portrait and the next section

### Border Radius Scale

| Radius | Use |
|--------|-----|
| 3–6px | Tiny decorative elements, micro-chips |
| 20px | Primary and secondary body CTAs |
| 24px | Consent/teal pill buttons, modal inner chips |
| 40px | Hero media frames, large section container corners |
| 50% | Circular portraits, icon-only buttons, satellite CTAs |
| 99px / 999px / 1000px | Full pill shapes — navigation, carousel cards, footer country selector |

## 6. Depth & Elevation

| Level | Treatment | Use |
|-------|-----------|-----|
| 0 | No shadow | The default — 95% of surfaces sit directly on white canvas |
| 1 | `rgba(10, 22, 40, 0.05) 0px 4px 24px 0px` | Floating nav pill — barely-there lift |
| 2 | `rgba(10, 22, 40, 0.08) 0px 24px 48px 0px` | Hero media frames, elevated cards |
| 3 | `rgba(10, 22, 40, 0.20) 0px 70px 110px 0px` | Rare; dramatic elevation on a feature tile |

### Shadow Philosophy
Shadows function as **atmospheric cushioning**, not directional light. The Level 2 shadow has a 48px spread and only 8% opacity — it barely exists as dark pixels but creates a "the card is breathing above the canvas" feel. No hard-edged, tight shadows anywhere. Border lines are preferred over shadows for functional delineation.

### Decorative Depth
- **Orbital arcs** (Sky Blue, ~1px): trace connective paths across sections
- **Ghost watermark headlines**: ice-blue-on-white text gives sections an almost-pressed-paper quality
- **Circle-image fade**: subjects at the edge of circular portraits dissolve into the white canvas

## 7. Do's and Don'ts

### Do
- Use Canvas White (`#FFFFFF`) as the default body background
- Use Lifted White (`#F7F9FC`) for nested or raised sections — the faint blue tint preserves the blue-white feel
- Mask service/feature imagery as perfect circles, not rectangles or rounded rectangles
- Attach a white satellite CTA to the bottom-right of each circular portrait
- Set headlines in weight 500 with -2% letter-spacing in Ocean Navy
- Use weight 450 (not 400) for body paragraphs
- Keep primary CTAs as Ocean Navy pills (20px radius) with white text
- Use Consent Teal only on consent, legal, or compliance actions
- Float the nav as a rounded white pill below the viewport top, not flush at y=0
- Build page rhythm from three surface tones: canvas white → lifted white → ocean navy footer
- Use thin Sky Blue arcs between service cards to imply connection

### Don't
- Don't use a warm cream or yellow-tinted background — it breaks the cool blue-white tone
- Don't round image frames at 8–16px — either use full-pill, 40px, or full-circle. In-between radii look generic
- Don't use Signal Blue for consent CTAs — reserve Consent Teal for legal color signals
- Don't mix typefaces — no serif accent, no script, no secondary display font
- Don't crowd the nav with more than six top-level links — the pill is meant to feel airy
- Don't drop hard shadows — all elevation should use 48px+ spread and ≤10% opacity
- Don't use uppercase for anything larger than the 14px eyebrow label
- Don't omit the tiny accent dot before eyebrow labels
- Don't place circular portraits on a grid — their magic comes from asymmetric placement

## 8. Responsive Behavior

### Breakpoints

| Name | Width | Key Changes |
|------|-------|-------------|
| Mobile | ≤ 767px | Nav pill shows logo + menu + search only; primary links hide behind hamburger; service portraits stack single-column centered; hero headline drops from 64px to ~40px; footer columns collapse into a vertical accordion |
| Tablet | 768–1023px | Nav pill shows 2–3 primary links truncated; service portraits arrange 2-up; hero headline ~48px |
| Desktop | ≥ 1024px | Full nav with 5 primary links centered; service portraits asymmetrically placed with decorative orbital lines; hero headline 64px |
| Wide | ≥ 1440px | Content max-width caps at ~1280px; gutters grow symmetrically; orbital lines extend further |

### Touch Targets
All interactive elements comfortably exceed 44×44px. The satellite CTA (circle + arrow) is ~50–60px. The nav pill buttons are ~48px tall. Mobile hamburger and search are 48×48px.

### Collapsing Strategy
- **Nav**: full pill → compact pill with hamburger. Pill shape is preserved across breakpoints.
- **Service grid**: asymmetric constellation → 2-up → 1-up stack. Orbital arcs are removed on mobile.
- **Spacing**: section vertical padding compresses from 128px to 48px on mobile.
- **Content**: two-column hero becomes stacked on mobile.
- **Footer**: 4 columns → 1 column accordion with chevron toggles.

## 9. Agent Prompt Guide

### Quick Color Reference
- Primary CTA: "Ocean Navy (`#0A1628`) — the deep navy used for primary pill buttons and footer"
- Background: "Canvas White (`#FFFFFF`) — bright white body canvas"
- Lifted surface: "Lifted White (`#F7F9FC`) — one step off-white with a faint blue tint for nested sections"
- Heading text: "Ocean Navy (`#0A1628`)"
- Body text: "Ocean Navy (`#0A1628`) at weight 450"
- Muted text: "Slate (`#536378`)"
- Signal / Interactive: "Signal Blue (`#0070D2`) — vivid action blue for interactive elements and decorative arcs"
- Consent / Legal: "Consent Teal (`#006E7A`) — reserve for cookie consent and legal actions only"
- Accent arc: "Sky Blue (`#5BA4E0`) — orbital decorative lines only"
- Border / Outline: "Ocean Navy at 1.5px for pill buttons; 1px at low opacity elsewhere"
- Footer: "Ocean Navy (`#0A1628`) with White text"

### Example Component Prompts
- "Create a circular portrait card 300px in diameter, with a square photograph cropped to a perfect circle. Attach a 56px white satellite button with a dark navy arrow icon at the bottom-right, so it protrudes ~40% outside the portrait. Below the portrait, add an eyebrow label with a Sky Blue dot and uppercase 'SERVICES' text in MarkForMC weight 700 at 14px. Below the eyebrow, set a 24px / weight 500 title in Ocean Navy."
- "Design a primary CTA button: Ocean Navy (`#0A1628`) background, White (`#FFFFFF`) text, 20px border-radius, 6px vertical and 24px horizontal padding, MarkForMC font at 16px weight 500 with -2% letter-spacing."
- "Build a floating navigation pill: white background with `rgba(10, 22, 40, 0.06) 0px 4px 24px 0px` shadow, 999px border-radius, ~16px vertical and 40px horizontal internal padding. Position it 24px below the viewport top, centered, with the brand logo at the left, five primary links centered with 48px gap, and a circular 48px search button at the right."
- "Create a hero media frame: 40px border-radius on all corners, full viewport width minus 48px gutters, ~60% viewport height, deep navy background for video content. Place it directly on the white canvas with no shadow."
- "Design a footer: Ocean Navy (`#0A1628`) background, white text, 4-column link grid with uppercase muted column headers at 14px weight 700 +4% tracking. Include a large conversational H2 above the grid, a 1px white-at-30%-opacity horizontal divider below, and a bottom row with copyright, legal small-print links, a pill-shaped country selector, and four social icons."

### Iteration Guide
When refining existing screens generated with this design system:
1. Focus on ONE component at a time
2. Reference specific color names AND hex codes from this document
3. Use natural language ("crisp white canvas", "stadium pill", "circular portrait with satellite CTA") alongside technical values
4. Describe the desired "feel" (clean, institutional, trustworthy) alongside specific measurements
5. When in doubt, reach for one of three radii: 20px (buttons), 40px (hero/stadium), or 999px (pill/nav)
6. Default backgrounds to Canvas White (`#FFFFFF`) — this single decision anchors the entire cool, professional mood

### Known Gaps
- The live page uses MarkForMC, a proprietary licensed typeface. Sofia Sans is the closest open-source substitute.
- Tablet breakpoint specifics (768–1023px) were inferred; intermediate layouts may vary per section.
- The exact "whisper" blue tone used for ghost-watermark headlines reads between `#EBF2FA` and `#D6E8F7`; the precise value varies per section.
- The Consent Teal (`#006E7A`) is the compliance signal and should not be confused with any marketing CTA color.
