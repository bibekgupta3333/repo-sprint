# Figma Design Prompt
# Sprint Intelligence Platform UI/UX

**Project**: LLM Agentic Sprint Intelligence Dashboard  
**Platform**: Web Application (Streamlit-based)  
**Target Users**: Project Managers, Developers at Small Startups  
**Design Goal**: Clean, data-dense, actionable insights

---

## Design System Prompt for AI Image Generation

Use this prompt to generate UI mockups in Figma, v0.dev, or with AI design tools (Midjourney, DALL-E, etc.).

### Master Prompt Template

```
Design a modern web application dashboard for sprint/milestone intelligence.

STYLE GUIDE:
- Design System: Material Design 3
- Color Palette: Professional SaaS (blues, greens, grays)
- Typography: Inter for UI, JetBrains Mono for code/metrics
- Layout: Responsive grid (12 columns), generous white space
- Data Visualization: Charts use Plotly-style gradients
- Shadows: Subtle elevation (4dp, 8dp, 16dp)
- Border Radius: 8px for cards, 4px for buttons
- Iconography: Heroicons or Material Icons

COMPONENTS TO DESIGN:
[Specify which screen - see sections below]

TARGET USERS:
- Project managers (non-technical, need clarity)
- Developers (technical, expect detail)
- Startup teams (limited time, need quick insights)

BRAND PERSONALITY:
- Trustworthy, intelligent, empowering
- Professional but approachable
- Data-driven, not overwhelming
```

---

## Color Palette (Industry Standard for SaaS Dashboards)

### Primary Colors
```css
/* Blues - Trust, Intelligence */
--primary-50:  #eff6ff;  /* Backgrounds */
--primary-100: #dbeafe;
--primary-200: #bfdbfe;
--primary-300: #93c5fd;
--primary-400: #60a5fa;
--primary-500: #3b82f6;  /* Primary buttons, links */
--primary-600: #2563eb;  /* Hover states */
--primary-700: #1d4ed8;
--primary-800: #1e40af;
--primary-900: #1e3a8a;
```

### Success/Warning/Error (Semantic Colors)
```css
/* Success - Green (sprints on track) */
--success-500: #10b981;
--success-600: #059669;

/* Warning - Amber (at risk) */
--warning-500: #f59e0b;
--warning-600: #d97706;

/* Error - Red (critical blockers) */
--error-500: #ef4444;
--error-600: #dc2626;

/* Info - Cyan (insights, recommendations) */
--info-500: #06b6d4;
--info-600: #0891b2;
```

### Neutral Colors
```css
/* Grays */
--gray-50:  #f9fafb;  /* Page background */
--gray-100: #f3f4f6;  /* Card backgrounds */
--gray-200: #e5e7eb;  /* Borders */
--gray-300: #d1d5db;
--gray-400: #9ca3af;  /* Disabled text */
--gray-500: #6b7280;  /* Secondary text */
--gray-600: #4b5563;  /* Body text */
--gray-700: #374151;
--gray-800: #1f2937;  /* Headings */
--gray-900: #111827;  /* Pure black text */
```

### Chart Colors (Data Visualization)
```css
/* Multi-series charts */
--chart-1: #3b82f6;  /* Blue */
--chart-2: #8b5cf6;  /* Purple */
--chart-3: #ec4899;  /* Pink */
--chart-4: #f59e0b;  /* Amber */
--chart-5: #10b981;  /* Green */
--chart-6: #06b6d4;  /* Cyan */
```

---

## Screen-by-Screen Design Prompts

### Screen 1: Dashboard (Organization Overview)

**Prompt**:
```
Design a SaaS dashboard homepage for sprint intelligence.

LAYOUT:
- Top: Navigation bar (logo left, user profile right)
- Sidebar: Navigation menu (Dashboard, Milestones, Dependencies, Recommendations, Settings)
- Main area: 3-column grid

TOP ROW (Metrics Cards - 3 columns):
1. "Active Sprints" - Large number "5", subtitle "3 on track, 2 at risk", green/amber icons
2. "Blockers Detected" - Large number "12", subtitle "4 critical", red exclamation icon
3. "Avg Health Score" - Large number "78%", circular progress bar, green

MIDDLE ROW (Charts - 2 columns):
[Left] Burndown chart (line chart, blue gradient):
- X-axis: Days (Feb 1 - Feb 28)
- Y-axis: Issues remaining
- 2 lines: "Ideal" (dashed gray) and "Actual" (solid blue)
- Current point highlighted

[Right] Sprint status distribution (donut chart):
- Green segment: "On Track" (60%)
- Amber segment: "At Risk" (30%)
- Red segment: "Off Track" (10%)
- Center: "10 Sprints"

BOTTOM ROW (Data Table):
Table header: "Active Milestones"
Columns: Repo | Milestone | Due Date | Progress | Health | Actions
5 rows of data with progress bars, health indicators (🟢🟡🔴)

COLOR SCHEME: Primary blues, semantic colors for status
STYLE: Clean, Material Design 3, subtle shadows
```

**Example AI Prompt (for Midjourney/DALL-E)**:
```
A modern web dashboard interface showing sprint analytics. Top section has 3 metric cards with large numbers and icons. Middle section shows a blue gradient burndown chart and a colorful donut chart. Bottom section has a data table with progress bars. Color palette: blues (#3b82f6), greens (#10b981), grays (#f3f4f6). Material Design style, clean, professional, SaaS product. --ar 16:9 --v 5
```

---

### Screen 2: Milestone Analysis (Deep Dive)

**Prompt**:
```
Design a detailed milestone analysis page.

HEADER:
- Breadcrumb: Dashboard > backend-api > Sprint 24 - Q1 2026
- Title: "Sprint 24 - Q1 2026"
- Subtitle: "Feb 1 - Feb 28, 2026 • 15 days remaining"
- Status badge: "At Risk" (amber, rounded pill)

LAYOUT (2-column):

LEFT COLUMN (60% width):
[Section 1] Prediction Summary Card:
- Large stat: "72% Completion Probability" (amber text)
- Sub-stats: "Likely date: Feb 26" | "3 days delayed"
- Confidence interval: "65% - 80%" (gray text)
- AI reasoning: Expandable text "Based on current velocity of 3.2 issues/day..."

[Section 2] Burndown Chart (interactive):
- Larger version of burndown from dashboard
- Hover tooltip showing specific dates/values
- Toggle buttons: "Ideal vs Actual" | "Issues vs Story Points"

[Section 3] Risk Assessment Table:
Columns: Risk | Severity | Probability | Impact | Affected
3 rows:
1. "Critical PR not merged" | "High" 🔴 | "85%" | "5 days" | #890, #1542, #1543
2. "Velocity decline" | "Medium" 🟡 | "60%" | "2 days" | N/A
3. "CI failures increasing" | "Low" 🟢 | "30%" | "1 day" | #1544

RIGHT COLUMN (40% width):
[Section 1] Health Score Gauge:
- Circular gauge: 78/100 (amber zone)
- Sub-metrics: Velocity, Coverage, Team Capacity

[Section 2] Team Activity:
- Avatar list of active contributors
- Contribution timeline (mini heatmap)

[Section 3] Recommendations (cards):
3 stacked cards:
1. "⚡ High Priority: Assign reviewer to PR #890"
   - Rationale: "Blocking 3 issues..."
   - Expected impact: "Reduce delay by 2-3 days"
   - Button: "View PR"

2. "📊 Medium: Reduce scope by 2 issues"
3. "👥 Low: Schedule team sync"

COLOR SCHEME: Blues for charts, amber for "at risk", green/amber/red for severity
STYLE: Information-dense but scannable, clear hierarchy
```

**AI Prompt Version**:
```
A detailed analytics page for project milestone. Left side: prediction card with percentage, burndown line chart, risk table. Right side: circular health gauge, team avatars, recommendation cards stacked vertically. Modern SaaS interface, blues and ambers, Material Design, professional. --ar 16:9
```

---

### Screen 3: Cross-Repo Dependency Graph

**Prompt**:
```
Design a dependency visualization page.

HEADER:
- Title: "Cross-Repository Dependencies"
- Filters: Dropdown "All Repos" | Date range picker

MAIN AREA:
Interactive force-directed graph (network visualization):

NODES:
- Repositories: Large circles (blue #3b82f6)
  - Label: "backend-api", "frontend-app", "mobile-app"
- Milestones: Medium circles (purple #8b5cf6)
  - Label: "Sprint 24", "Sprint 25"
- Issues: Small circles (color by status)
  - Green: Closed
  - Amber: Open on-track
  - Red: Blocked

EDGES:
- Solid lines: Direct dependency (issue blocks issue)
- Dashed lines: Indirect dependency (shared contributor)
- Line width: Dependency strength

INTERACTION:
- Hover node: Highlight connected nodes, show tooltip
- Click node: Right panel appears with details

RIGHT PANEL (20% width, collapsible):
[When issue clicked]
- Issue title: "#1542: Fix memory leak"
- Status: "Blocked" 🔴
- Blocks: "#1543, #1544, #1545"
- Blocked by: "PR #890 (not merged)"
- Button: "View Issue"

LEGEND (bottom-right):
- Node types (Repo, Milestone, Issue)
- Edge types (Direct, Indirect)
- Colors (Status meanings)

COLOR SCHEME: Blues for repos, purples for milestones, status colors for issues
STYLE: Dark mode-friendly graph, light background, Figma-like node style
```

**AI Prompt**:
```
Network graph visualization showing repository dependencies. Large blue circles for repos, medium purple circles for milestones, small colored circles for issues. Lines connecting them. Clean white background, modern graph visualization style, similar to Figma or Miro. Right sidebar panel. --ar 16:9
```

---

### Screen 4: Recommendations / Action Items

**Prompt**:
```
Design a recommendation feed page.

HEADER:
- Title: "AI Recommendations"
- Filters: "All" | "High Priority" | "Accepted" | "Dismissed"
- Sort: "Priority" dropdown

LAYOUT:
Vertical feed of recommendation cards (list view):

CARD STRUCTURE (repeating):
┌──────────────────────────────────────────────────────┐
│ 🔴 CRITICAL PRIORITY                          [Expand]│
│ Assign additional reviewer to PR #890                │
│                                                       │
│ 📊 RATIONALE:                                         │
│ PR #890 has been open for 5 days and is blocking     │
│ 3 high-priority issues (#1542, #1543, #1544).        │
│ Based on 23 similar cases, adding a reviewer         │
│ reduces merge time by 40%.                           │
│                                                       │
│ 🎯 EXPECTED IMPACT:                                   │
│ • Reduce delay by 2-3 days                           │
│ • Unblock 3 downstream issues                        │
│                                                       │
│ 💡 EVIDENCE:                                          │
│ • PR #890 (review pending for 5 days)                │
│ • Similar case: PR #712 → Merged in 2 days after     │
│   reviewer added                                     │
│                                                       │
│ ⚙️ EFFORT: Medium (2 hours) | CONFIDENCE: 91%        │
│                                                       │
│ [Accept] [Dismiss] [View PR #890]                    │
└──────────────────────────────────────────────────────┘

[Repeat 4-5 cards with varying priority colors]

PRIORITY COLORS:
- Critical: Red border-left (#ef4444)
- High: Amber border-left (#f59e0b)
- Medium: Blue border-left (#3b82f6)
- Low: Gray border-left (#6b7280)

INTERACTION:
- Click "Expand" → Show full evidence list
- Click "Accept" → Mark green, track outcome
- Click "Dismiss" → Hide card, optional feedback modal

EMPTY STATE (if no recommendations):
- Illustration: Checkmark icon
- Text: "All caught up! No critical recommendations."
- Subtitle: "We'll notify you when issues arise."

COLOR SCHEME: Semantic colors (red, amber, blue, gray), clean white cards
STYLE: Card-based feed, similar to GitHub notifications
```

**AI Prompt**:
```
Recommendation feed interface with stacked cards. Each card has colored left border (red, amber), title, expandable sections (rationale, impact, evidence), and action buttons. Clean white cards on light gray background. Modern SaaS style. --ar 9:16
```

---

### Screen 5: Settings / Configuration

**Prompt**:
```
Design a settings page.

LAYOUT: Tabbed interface

TABS:
1. Repositories
2. Notifications
3. Integrations
4. Preferences

TAB 1: REPOSITORIES
┌─────────────────────────────────────────────────┐
│ Connected Repositories                          │
│                                                 │
│ [+ Add Repository]                              │
│                                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │ ✓ backend-api                        [Remove]│ │
│ │   microsoft/backend-api                      │ │
│ │   Last synced: 2 minutes ago                 │ │
│ │   [Sync Now]                                 │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │ ✓ frontend-app                       [Remove]│ │
│ │   microsoft/frontend-app                     │ │
│ │   Last synced: 5 minutes ago                 │ │
│ │   [Sync Now]                                 │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ [Add Repository] button:                        │
│ → Opens modal with GitHub OAuth flow            │
└─────────────────────────────────────────────────┘

TAB 2: NOTIFICATIONS
Toggle switches:
☑ Email notifications for critical blockers
☑ Slack integration (channel: #sprints)
☐ Daily digest email
☑ Real-time dashboard updates

TAB 3: INTEGRATIONS
Cards for:
- GitHub (Connected ✓)
- Slack (Configure)
- Jira (Coming Soon)

TAB 4: PREFERENCES
- Theme: Light | Dark | Auto
- Date format: MM/DD/YYYY | DD/MM/YYYY
- Time zone: (UTC-8) Pacific Time

COLOR SCHEME: Blues for active states, grays for neutral
STYLE: Clean settings page, toggle switches, form inputs
```

---

## Component Library (Figma-Style Spec)

### Buttons
```
Primary Button:
- Background: --primary-500 (#3b82f6)
- Text: White, 14px, Medium weight
- Padding: 12px 24px
- Border Radius: 8px
- Hover: --primary-600 (#2563eb)
- Active: --primary-700 (#1d4ed8)

Secondary Button:
- Background: Transparent
- Border: 1px solid --gray-300 (#d1d5db)
- Text: --gray-700 (#374151), 14px, Medium
- Padding: 12px 24px
- Border Radius: 8px
- Hover: --gray-100 background (#f3f4f6)
```

### Cards
```
Default Card:
- Background: White
- Border: 1px solid --gray-200 (#e5e7eb)
- Border Radius: 12px
- Shadow: 0 1px 3px rgba(0,0,0,0.1)
- Padding: 24px

Hover Card:
- Shadow: 0 4px 6px rgba(0,0,0,0.1)
```

### Status Badges
```
Badge (pill-shaped):
- Border Radius: 16px
- Padding: 4px 12px
- Font: 12px, Medium weight

Variants:
- On Track: bg --success-100, text --success-700
- At Risk: bg --warning-100, text --warning-700
- Off Track: bg --error-100, text --error-700
```

### Typography
```
H1 (Page Title): 32px, Bold, --gray-900
H2 (Section): 24px, Semibold, --gray-800
H3 (Card Title): 18px, Semibold, --gray-800
Body: 14px, Regular, --gray-600
Caption: 12px, Regular, --gray-500
Code/Metrics: JetBrains Mono, 14px, Monospace
```

---

## Responsive Breakpoints

```
Mobile: < 768px (single column, simplified charts)
Tablet: 768px - 1024px (2 columns, stacked cards)
Desktop: 1024px - 1440px (3 columns, full layout)
Wide: > 1440px (3-4 columns, expanded sidebars)
```

---

## Accessibility Requirements

- **Color Contrast**: WCAG AA compliance (4.5:1 for text)
- **Focus States**: 2px blue outline on keyboard focus
- **Alt Text**: All charts have text alternatives
- **Keyboard Navigation**: All actions accessible via keyboard
- **Screen Reader**: ARIA labels on interactive elements

---

## Animation Guidelines

```
Transitions: 200ms ease-in-out
Page Loads: Skeleton screens (gray placeholders)
Data Updates: Smooth number increments (CountUp.js style)
Charts: Staggered animation on load
Modals: Fade in + scale (0.95 → 1.0)
```

---

## Export Specifications

For Figma → Streamlit implementation:

1. **Colors**: Export as CSS variables (see palette above)
2. **Typography**: Export font scales
3. **Components**: Build reusable Streamlit components
4. **Icons**: Use Heroicons or Material Icons (24px baseline)
5. **Spacing**: 8px grid system (8px, 16px, 24px, 32px, 48px)

---

## Example Figma File Structure

```
📁 Sprint Intelligence UI Kit
├── 📄 Cover Page
├── 🎨 Design System
│   ├── Colors
│   ├── Typography
│   ├── Components
│   └── Icons
├── 📱 Screens
│   ├── 1. Dashboard
│   ├── 2. Milestone Analysis
│   ├── 3. Dependencies
│   ├── 4. Recommendations
│   └── 5. Settings
└── 📐 Responsive Layouts
    ├── Mobile
    ├── Tablet
    └── Desktop
```

---

**Deliverable**: Use these prompts to generate UI mockups, then implement with Streamlit + Plotly.

**Tools Recommended**:
- **Figma**: For detailed design system
- **v0.dev**: AI-generated React components (adapt to Streamlit)
- **Midjourney/DALL-E**: Quick concept mockups
- **Streamlit**: Final implementation

**Document Version**: 1.0.0  
**Last Updated**: February 14, 2026
