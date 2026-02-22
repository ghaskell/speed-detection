# Design System — Traffic Monitoring System

Municipal/government utility aesthetic. Dense, data-forward, utilitarian.
Uses **only** standard Tailwind CSS utility classes. No custom config, no CSS variables, no `@apply`.

Tailwind is loaded via CDN with `darkMode: 'class'` strategy.

```html
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config = { darkMode: 'class' }</script>
```

---

## 1. Typography

System font stack via `font-sans`. Data values and log output use `font-mono`.

| Role | Tailwind Classes |
|------|-----------------|
| Page title | `text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-100` |
| Section heading | `text-base font-semibold text-slate-800 dark:text-slate-200` |
| Subsection heading | `text-sm font-semibold text-slate-700 dark:text-slate-300` |
| Body text | `text-sm text-slate-700 dark:text-slate-300` |
| Labels | `text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400` |
| Data values | `font-mono text-sm text-slate-900 dark:text-slate-100` |
| Large KPI number | `font-mono text-2xl font-bold` (+ status color) |
| Small/meta text | `text-xs text-slate-500 dark:text-slate-400` |
| Table header | `text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400` |
| Table cell | `text-sm font-mono text-slate-800 dark:text-slate-200` |
| Nav link | `text-sm font-medium` |
| Log message | `font-mono text-xs text-slate-700 dark:text-slate-300` |

**Rules:**
- Body text is always `text-sm`. Dense areas use `text-xs`.
- Comparable data (speeds, counts, timestamps) always uses `font-mono`.
- No italic. Bold only for headings, labels, and danger emphasis.

---

## 2. Color Palette

### Surfaces

| Surface | Light | Dark |
|---------|-------|------|
| Page background | `bg-slate-100` | `dark:bg-slate-900` |
| Card / panel | `bg-white` | `dark:bg-slate-800` |
| Secondary panel | `bg-slate-50` | `dark:bg-slate-800` |
| Inset / well | `bg-slate-50` | `dark:bg-slate-950` |
| Nav bar | `bg-white` | `dark:bg-slate-800` |
| Table header row | `bg-slate-50` | `dark:bg-slate-800` |
| Row hover | `hover:bg-slate-50` | `dark:hover:bg-slate-700` |

### Text

| Role | Light | Dark |
|------|-------|------|
| Primary | `text-slate-900` | `dark:text-slate-100` |
| Secondary | `text-slate-600` | `dark:text-slate-400` |
| Muted | `text-slate-400` | `dark:text-slate-500` |

### Borders

| Type | Light | Dark |
|------|-------|------|
| Default | `border-slate-200` | `dark:border-slate-700` |
| Strong | `border-slate-300` | `dark:border-slate-600` |
| Input focus | `focus:border-blue-500` | `dark:focus:border-blue-400` |

### Status Colors

| Status | Badge bg | Badge text | Value text |
|--------|----------|------------|------------|
| Info | `bg-blue-100 dark:bg-blue-900` | `text-blue-700 dark:text-blue-300` | `text-blue-600 dark:text-blue-400` |
| Success | `bg-emerald-100 dark:bg-emerald-900` | `text-emerald-700 dark:text-emerald-300` | `text-emerald-600 dark:text-emerald-400` |
| Warning | `bg-amber-100 dark:bg-amber-900` | `text-amber-700 dark:text-amber-300` | `text-amber-600 dark:text-amber-400` |
| Danger | `bg-red-100 dark:bg-red-900` | `text-red-700 dark:text-red-300` | `text-red-600 dark:text-red-400` |
| Neutral | `bg-slate-100 dark:bg-slate-700` | `text-slate-500 dark:text-slate-400` | `text-slate-500 dark:text-slate-400` |

### Traffic Light Indicators

| State | Active | Inactive |
|-------|--------|----------|
| Red | `bg-red-500` | `bg-slate-300 dark:bg-slate-600` |
| Yellow | `bg-amber-400` | `bg-slate-300 dark:bg-slate-600` |
| Green | `bg-emerald-500` | `bg-slate-300 dark:bg-slate-600` |

---

## 3. Spacing

Dense layout. Err on tight spacing.

| Context | Value |
|---------|-------|
| Page padding | `px-4 py-4` |
| Container max-width | `max-w-7xl mx-auto` (default), `max-w-screen-xl` (calibration), `max-w-3xl` (forms) |
| Card padding | `p-4` or `px-3 py-3` (compact) |
| Section spacing | `space-y-4` or `mb-4` |
| Grid gap (stats) | `gap-3` |
| Grid gap (gallery) | `gap-4` |
| Form field padding | `px-3 py-2` |
| Label to input | `mb-1` |
| Nav bar | `px-4 py-2`, `gap-2` |
| Table cell | `px-3 py-2` |
| Badge | `px-2 py-0.5` |
| Button | `px-3 py-1.5` (small), `px-4 py-2` (default) |

---

## 4. Borders & Radius

Municipal = squared off. Minimal rounding. **No shadows.**

| Element | Border | Radius |
|---------|--------|--------|
| Card / panel | `border border-slate-200 dark:border-slate-700` | `rounded` |
| Input | `border border-slate-300 dark:border-slate-600` | `rounded` |
| Button (filled) | none | `rounded` |
| Button (outline) | `border border-slate-300 dark:border-slate-600` | `rounded` |
| Badge | none | `rounded` (NOT `rounded-full`) |
| Table wrapper | `border border-slate-200 dark:border-slate-700` | `rounded` |
| Image / video | `border-2 border-slate-300 dark:border-slate-600` | `rounded` |
| Nav bar | `border-b border-slate-200 dark:border-slate-700` | none |

---

## 5. Component Patterns

### 5.1 Buttons

```
Primary:   px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium
Secondary: px-4 py-2 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-sm font-medium text-slate-700 dark:text-slate-300
Danger:    px-4 py-2 rounded bg-red-600 hover:bg-red-700 text-white text-sm font-medium
Small:     px-2 py-1 rounded bg-red-600 hover:bg-red-700 text-white text-xs font-medium
Disabled:  px-4 py-2 rounded bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 text-sm font-medium cursor-not-allowed
Ghost:     px-3 py-1.5 rounded text-sm font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700
```

### 5.2 Badges

```
Base: inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold
```
Append status color pair from table above.

### 5.3 Form Inputs

```
Input:  w-full px-3 py-2 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:border-blue-500 dark:focus:border-blue-400 focus:ring-1 focus:ring-blue-500 dark:focus:ring-blue-400 focus:outline-none
Label:  block text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-1
Select: (same as Input)
```

### 5.4 Stat Card

```html
<div class="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-3 py-3 text-center">
  <div class="font-mono text-2xl font-bold text-slate-900 dark:text-slate-100">1,247</div>
  <div class="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400 mt-1">Total Vehicles</div>
</div>
```

### 5.5 Data Table

```html
<div class="border border-slate-200 dark:border-slate-700 rounded overflow-hidden">
  <div class="overflow-x-auto">
    <table class="w-full text-left">
      <thead>
        <tr class="bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
          <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Column</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-200 dark:divide-slate-700">
        <tr class="hover:bg-slate-50 dark:hover:bg-slate-700">
          <td class="px-3 py-2 font-mono text-sm text-slate-800 dark:text-slate-200">Value</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
```

### 5.6 Image Gallery Card

```html
<div class="border border-slate-200 dark:border-slate-700 rounded overflow-hidden bg-white dark:bg-slate-800">
  <img class="w-full h-44 object-cover border-b border-slate-200 dark:border-slate-700">
  <div class="px-3 py-2">
    <div class="font-mono text-lg font-bold text-red-600 dark:text-red-400">47 mph</div>
    <div class="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Lane 1 · 14:32</div>
  </div>
</div>
```

### 5.7 Filter Chips

```
Active:   px-2.5 py-1 rounded border text-xs font-medium bg-blue-100 dark:bg-blue-900 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300
Inactive: px-2.5 py-1 rounded border text-xs font-medium bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700
```

### 5.8 Mode Toggle Buttons

```
Active:   flex-1 px-3 py-2 rounded border-2 border-blue-500 bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300 text-sm font-medium text-center
Inactive: flex-1 px-3 py-2 rounded border-2 border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-900 text-slate-500 dark:text-slate-400 text-sm font-medium text-center hover:border-slate-400 dark:hover:border-slate-500
```

### 5.9 Day Selector Buttons

```
Active:   px-3 py-1.5 rounded border text-sm font-medium bg-blue-600 border-blue-600 text-white
Inactive: px-3 py-1.5 rounded border text-sm font-medium bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800
```

---

## 6. Dark / Light Mode

- Class-based toggling on `<html>` element
- Default: dark mode (monitoring app)
- Persisted in `localStorage` key `theme`
- Init script runs in `<head>` before body paint to prevent flash
- Toggle button in nav bar (sun/moon icons)

**Rules:**
1. Every `bg-*` must have a `dark:bg-*` counterpart
2. Every `text-*` color must have a `dark:text-*` counterpart
3. Every `border-*` color must have a `dark:border-*` counterpart
4. Never use `text-white` or `text-black` for content — use the slate scale

---

## 7. Responsive Breakpoints

| Breakpoint | Usage |
|-----------|-------|
| Default | Single column, stacked |
| `sm:` (640px) | 2-col stat grids, side-by-side form rows |
| `lg:` (1024px) | 3-col galleries, sidebar layouts |
| `xl:` (1280px) | 4-col galleries, full stat rows |

---

## 8. JS Class Toggling Pattern

When JavaScript toggles visual states, define Tailwind class strings as constants:

```javascript
const STATE_ACTIVE = 'px-3 py-1.5 rounded border bg-blue-600 border-blue-600 text-white text-sm font-medium';
const STATE_INACTIVE = 'px-3 py-1.5 rounded border bg-white dark:bg-slate-900 border-slate-300 dark:border-slate-600 text-slate-500 text-sm font-medium';

// Toggle:
element.className = isActive ? STATE_ACTIVE : STATE_INACTIVE;
```

Never define custom CSS class names for JS to reference. Always use full Tailwind class strings.

---

## 9. Iconography

No icon library. HTML entities only:

| Symbol | Entity | Usage |
|--------|--------|-------|
| &larr; | `&larr;` | Back navigation |
| &times; | `&times;` | Delete / close |
| &#9788; | `&#9788;` | Sun (light mode) |
| &#9789; | `&#9789;` | Moon (dark mode) |
| &middot; | `&middot;` | Metadata separator |
