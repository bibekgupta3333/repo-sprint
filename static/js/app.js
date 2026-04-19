/* ═══════════════════════════════════════════════════════════════════════
   Sprint Intelligence — Frontend Application
   ═══════════════════════════════════════════════════════════════════════ */

const STORAGE_KEYS = {
  activeFilters: 'sprint-ui.active-filters.v1',
  filterPresets: 'sprint-ui.filter-presets.v1',
  runNotes: 'sprint-ui.run-notes.v1',
  teamAnnotations: 'sprint-ui.team-annotations.v1',
  theme: 'sprint-ui.theme.v1',
};

const DEFAULT_VIEW_FILTERS = {
  organization: '',
  dateRange: 'all',
  riskThreshold: 'all',
  sourceMode: 'all',
};

const DASHBOARD_PAGES = ['dashboard', 'analyze', 'results', 'dependencies', 'recommendations', 'ingestion', 'settings', 'about'];
const DRILLDOWN_DEFAULT_SORT = { key: 'z_delta', direction: 'desc' };
const SOURCE_RELIABILITY_SCORES = {
  llm: 1,
  fallback: 0.65,
  hybrid: 0.78,
  rule_based: 0.55,
  error: 0.2,
};

// ── State ──
const AppState = {
  currentPage: 'dashboard',
  inputMode: 'json',
  lastResult: null,
  isRunning: false,
  sprintFiles: [],
  organizationIndex: [],
  selectedOrganization: '',
  selectedRunId: '',
  selectedOrgRuns: [],
  comparisonSelection: [],
  comparisonResults: [],
  timeline: {
    org: '',
    runs: [],
    currentIndex: 0,
    playing: false,
    timer: null,
    runDetailsCache: {},
  },
  alerts: [],
  filters: { ...DEFAULT_VIEW_FILTERS },
  filterPresets: {},
  runNotes: {},
  teamAnnotations: {},
  drilldown: {
    query: '',
    sortKey: DRILLDOWN_DEFAULT_SORT.key,
    sortDir: DRILLDOWN_DEFAULT_SORT.direction,
  },
  evidenceScorecard: null,
  deepLink: {
    isApplying: false,
    pendingRunId: '',
  },
  commandPalette: {
    open: false,
    query: '',
    selectedIndex: 0,
    commands: [],
  },
  keyboard: {
    cardFocusIndex: -1,
  },
  settings: {
    autoRefresh: false,
    refreshInterval: 30,
    evalMode: 'resilient',
    showRawJson: false,
    theme: 'dark',
  },
  ingestion: {
    running: false,
  },
  charts: {},
  dashboardCharts: {},
  models: {
    available: [],
    current: null,
    selected: null,
  },
};

function getStoredTheme() {
  try {
    const storedTheme = localStorage.getItem(STORAGE_KEYS.theme);
    if (storedTheme === 'light' || storedTheme === 'dark') {
      return storedTheme;
    }
  } catch (_) {
    // Ignore localStorage access errors.
  }
  return '';
}

function getSystemThemePreference() {
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    return 'light';
  }
  return 'dark';
}

function updateThemeControls(theme) {
  const toggleButton = document.getElementById('theme-toggle-btn');
  if (toggleButton) {
    const label = theme === 'light' ? '☀️ Light' : '🌙 Dark';
    toggleButton.textContent = label;
    toggleButton.setAttribute('title', `Switch to ${theme === 'light' ? 'dark' : 'light'} mode`);
    toggleButton.setAttribute('aria-label', `Current theme: ${theme}. Click to switch theme`);
    toggleButton.setAttribute('aria-pressed', theme === 'light' ? 'true' : 'false');
  }

  const toggleSwitch = document.getElementById('theme-toggle-switch');
  if (toggleSwitch) {
    toggleSwitch.checked = theme === 'light';
  }
}

function applyTheme(theme, options = {}) {
  const { persist = true, notify = false } = options;
  const normalizedTheme = theme === 'light' ? 'light' : 'dark';

  document.documentElement.setAttribute('data-theme', normalizedTheme);
  document.documentElement.style.colorScheme = normalizedTheme;
  AppState.settings.theme = normalizedTheme;

  if (persist) {
    try {
      localStorage.setItem(STORAGE_KEYS.theme, normalizedTheme);
    } catch (_) {
      // Ignore localStorage access errors.
    }
  }

  updateThemeControls(normalizedTheme);

  if (notify) {
    showToast(`${normalizedTheme === 'light' ? 'Light' : 'Dark'} mode enabled`, 'info');
  }
}

function initTheme() {
  const domTheme = document.documentElement.getAttribute('data-theme');
  const initialTheme = (domTheme === 'light' || domTheme === 'dark')
    ? domTheme
    : (getStoredTheme() || getSystemThemePreference());

  applyTheme(initialTheme, { persist: false, notify: false });
}

function toggleTheme(forceTheme = '') {
  const nextTheme = (forceTheme === 'light' || forceTheme === 'dark')
    ? forceTheme
    : (AppState.settings.theme === 'light' ? 'dark' : 'light');

  applyTheme(nextTheme, { persist: true, notify: true });
}

// ── Agent pipeline stages ──
const STAGES = [
  { name: 'Router',           icon: '🔀', desc: 'Validating input & preparing state…' },
  { name: 'Data Collector',   icon: '📥', desc: 'Fetching GitHub issues, PRs, commits…' },
  { name: 'Dep Graph',        icon: '🔗', desc: 'Mapping cross-repo dependencies…' },
  { name: 'Feature Eng.',     icon: '⚙️', desc: 'Extracting 18 metrics across 5 categories…' },
  { name: 'Synthetic Gen.',   icon: '🧬', desc: 'Generating cold-start bootstrapping data…' },
  { name: 'Embedding RAG',    icon: '🧲', desc: 'Building RAG context via ChromaDB…' },
  { name: 'LLM Reasoner',     icon: '🤖', desc: 'Predicting completion probability via Ollama…' },
  { name: 'Sprint Analyzer',  icon: '📈', desc: 'Computing composite health score…' },
  { name: 'LoRA Training',    icon: '🎓', desc: 'Continuous learning — adapter check…' },
  { name: 'Risk Assessor',    icon: '🛡️', desc: 'Identifying blockers & velocity gaps…' },
  { name: 'Recommender',      icon: '💡', desc: 'Generating interventions from precedent…' },
  { name: 'Explainer',        icon: '📝', desc: 'Writing evidence-backed narrative report…' },
];

const EXAMPLE_SPRINT_INPUT = {
  sprint_id: 'example_sprint_001',
  repo: 'Mintplex-Labs/anything-llm',
  start_date: '2026-03-01T00:00:00Z',
  end_date: '2026-03-14T23:59:59Z',
  issues: [
    {
      number: 101,
      title: 'Stabilize authentication refresh flow',
      body: 'Intermittent session drops under concurrent API calls. Must complete before release candidate.',
      state: 'closed',
      labels: ['backend', 'auth', 'priority-high'],
      created_at: '2026-03-01T09:12:00Z',
      updated_at: '2026-03-03T11:45:00Z',
      closed_at: '2026-03-03T11:45:00Z',
      author: 'alice',
      url: 'https://github.com/Mintplex-Labs/anything-llm/issues/101',
      related_prs: [201],
    },
    {
      number: 102,
      title: 'Improve vector search latency for dashboard query mode',
      body: 'Search endpoint exceeds P95 target in peak ingestion windows.',
      state: 'open',
      labels: ['performance', 'rag', 'priority-medium'],
      created_at: '2026-03-02T10:05:00Z',
      updated_at: '2026-03-10T14:20:00Z',
      author: 'bob',
      url: 'https://github.com/Mintplex-Labs/anything-llm/issues/102',
      related_prs: [202],
    },
    {
      number: 103,
      title: 'Add explainability panel evidence grouping',
      body: 'Group citations by issue/PR/commit so non-technical stakeholders can review faster.',
      state: 'open',
      labels: ['frontend', 'ux', 'explainability'],
      created_at: '2026-03-04T13:30:00Z',
      updated_at: '2026-03-11T08:55:00Z',
      author: 'carol',
      url: 'https://github.com/Mintplex-Labs/anything-llm/issues/103',
      related_prs: [203],
    },
  ],
  pull_requests: [
    {
      number: 201,
      title: 'Fix token refresh race condition',
      body: 'Adds lock-based refresh guard and retries for transient auth provider errors.',
      state: 'closed',
      labels: ['auth', 'bugfix'],
      author: 'alice',
      additions: 128,
      deletions: 36,
      commits: 4,
      created_at: '2026-03-02T07:40:00Z',
      updated_at: '2026-03-03T11:40:00Z',
      closed_at: '2026-03-03T11:40:00Z',
      merged_at: '2026-03-03T11:40:00Z',
      url: 'https://github.com/Mintplex-Labs/anything-llm/pull/201',
      file_diffs: [
        { filename: 'src/auth/session.ts', status: 'modified', additions: 74, deletions: 19 },
        { filename: 'src/auth/token.ts', status: 'modified', additions: 54, deletions: 17 },
      ],
    },
    {
      number: 202,
      title: 'Optimize retrieval query batching',
      body: 'Introduces request coalescing and adaptive chunk sizes for lower latency.',
      state: 'open',
      labels: ['performance', 'rag'],
      author: 'david',
      additions: 212,
      deletions: 61,
      commits: 6,
      created_at: '2026-03-05T08:25:00Z',
      updated_at: '2026-03-12T16:50:00Z',
      url: 'https://github.com/Mintplex-Labs/anything-llm/pull/202',
      file_diffs: [
        { filename: 'src/retrieval/query_runner.py', status: 'modified', additions: 147, deletions: 44 },
        { filename: 'src/retrieval/cache.py', status: 'modified', additions: 65, deletions: 17 },
      ],
    },
  ],
  commits: [
    {
      sha: 'a1b2c3d',
      sha_full: 'a1b2c3d4e5f67890aabbccddeeff001122334455',
      message: 'fix(auth): guard refresh flow under concurrent load',
      body: 'Prevents duplicate refresh calls and session invalidation race.',
      author: 'alice',
      created_at: '2026-03-03T11:10:00Z',
      url: 'https://github.com/Mintplex-Labs/anything-llm/commit/a1b2c3d4e5f67890aabbccddeeff001122334455',
      diff: {
        total_additions: 45,
        total_deletions: 12,
        files_changed: 3,
        file_diffs: [
          { filename: 'src/auth/session.ts', status: 'modified', additions: 22, deletions: 5 },
          { filename: 'src/auth/token.ts', status: 'modified', additions: 17, deletions: 4 },
          { filename: 'tests/auth/session.test.ts', status: 'modified', additions: 6, deletions: 3 },
        ],
      },
    },
    {
      sha: 'd4e5f6a',
      sha_full: 'd4e5f6a7b8c90123ddeeffaabb11223344556677',
      message: 'perf(rag): coalesce vector queries for dashboard latency',
      body: 'Bundles near-simultaneous requests and improves chunk planning.',
      author: 'david',
      created_at: '2026-03-10T15:42:00Z',
      url: 'https://github.com/Mintplex-Labs/anything-llm/commit/d4e5f6a7b8c90123ddeeffaabb11223344556677',
      diff: {
        total_additions: 62,
        total_deletions: 18,
        files_changed: 4,
        file_diffs: [
          { filename: 'src/retrieval/query_runner.py', status: 'modified', additions: 35, deletions: 10 },
          { filename: 'src/retrieval/batching.py', status: 'modified', additions: 18, deletions: 5 },
          { filename: 'src/retrieval/cache.py', status: 'modified', additions: 7, deletions: 2 },
          { filename: 'tests/retrieval/query_runner.test.py', status: 'modified', additions: 2, deletions: 1 },
        ],
      },
    },
  ],
};

let animTimer = null;
let stageIdx = 0;

// ═══════════════════════════════════════════════════════════════
// Navigation
// ═══════════════════════════════════════════════════════════════

function navigateTo(page) {
  AppState.currentPage = page;

  // Update nav
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.page === page);
  });

  // Update page views
  document.querySelectorAll('.page-view').forEach(el => {
    el.classList.toggle('active', el.id === `page-${page}`);
  });

  // Update header
  const titles = {
    dashboard:      ['Dashboard',        'Multi-agent sprint intelligence overview'],
    analyze:        ['Sprint Analysis',  'Configure and run agent inference pipeline'],
    results:        ['Analysis Results', 'Detailed breakdown of sprint health assessment'],
    dependencies:   ['Dependencies',     'Cross-repository dependency intelligence'],
    recommendations:['Interventions',    'AI-generated actionable recommendations'],
    ingestion:      ['Ingestion',        'Run organization data ingestion pipeline from dashboard'],
    settings:       ['Settings',         'System configuration and preferences'],
    about:          ['About System',     'Architecture, orchestration, and pipeline details'],
  };
  const [title, desc] = titles[page] || ['Sprint Intelligence', ''];
  document.getElementById('page-title').textContent = title;
  document.getElementById('page-desc').textContent = desc;

  if (!AppState.lastResult) {
    renderContextualEmptyState(page);
  }

  setTimeout(() => refreshFocusableCards(), 0);
  syncDeepLinkState();
}

// ═══════════════════════════════════════════════════════════════
// Toasts
// ═══════════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    toast.style.transition = 'all .3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatRelativeTime(isoTime) {
  if (!isoTime) return 'unknown';
  const target = new Date(isoTime);
  if (Number.isNaN(target.getTime())) return isoTime;

  const diffMs = Date.now() - target.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;

  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;

  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay}d ago`;

  return target.toLocaleDateString();
}

function asNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function setInputMode(mode) {
  AppState.inputMode = mode === 'query' ? 'query' : 'json';

  const jsonTab = document.getElementById('tab-input-json');
  const queryTab = document.getElementById('tab-input-query');
  const jsonPane = document.getElementById('json-input-pane');
  const queryPane = document.getElementById('query-input-pane');
  const modeLabel = document.getElementById('input-mode-label');

  if (jsonTab && queryTab) {
    jsonTab.classList.toggle('active', AppState.inputMode === 'json');
    queryTab.classList.toggle('active', AppState.inputMode === 'query');
  }

  if (jsonPane && queryPane) {
    jsonPane.classList.toggle('input-pane-hidden', AppState.inputMode !== 'json');
    queryPane.classList.toggle('input-pane-hidden', AppState.inputMode !== 'query');
  }

  if (modeLabel) {
    modeLabel.textContent = AppState.inputMode === 'json' ? 'JSON Input' : 'Query Text';
  }

  syncDeepLinkState();
}

function stopTimelinePlayback() {
  if (AppState.timeline.timer) {
    clearInterval(AppState.timeline.timer);
    AppState.timeline.timer = null;
  }
  AppState.timeline.playing = false;
}

function safeJsonParse(raw, fallback) {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch (_) {
    return fallback;
  }
}

function normalizeFilterModel(model = {}) {
  return {
    organization: typeof model.organization === 'string' ? model.organization : '',
    dateRange: typeof model.dateRange === 'string' ? model.dateRange : 'all',
    riskThreshold: typeof model.riskThreshold === 'string' ? model.riskThreshold : 'all',
    sourceMode: typeof model.sourceMode === 'string' ? model.sourceMode : 'all',
  };
}

function loadPersistedViewState() {
  const active = safeJsonParse(localStorage.getItem(STORAGE_KEYS.activeFilters), {});
  const presets = safeJsonParse(localStorage.getItem(STORAGE_KEYS.filterPresets), {});
  const notes = safeJsonParse(localStorage.getItem(STORAGE_KEYS.runNotes), {});
  const annotations = safeJsonParse(localStorage.getItem(STORAGE_KEYS.teamAnnotations), {});

  AppState.filters = { ...DEFAULT_VIEW_FILTERS, ...normalizeFilterModel(active) };
  AppState.filterPresets = presets && typeof presets === 'object' ? presets : {};
  AppState.runNotes = notes && typeof notes === 'object' ? notes : {};
  AppState.teamAnnotations = annotations && typeof annotations === 'object' ? annotations : {};
}

function persistActiveFilters() {
  localStorage.setItem(STORAGE_KEYS.activeFilters, JSON.stringify(AppState.filters));
}

function persistFilterPresets() {
  localStorage.setItem(STORAGE_KEYS.filterPresets, JSON.stringify(AppState.filterPresets));
}

function persistRunNotes() {
  localStorage.setItem(STORAGE_KEYS.runNotes, JSON.stringify(AppState.runNotes));
}

function persistTeamAnnotations() {
  localStorage.setItem(STORAGE_KEYS.teamAnnotations, JSON.stringify(AppState.teamAnnotations));
}

function getCurrentRunMeta() {
  const timelineMeta = getTimelineCurrentRunMeta();
  if (timelineMeta && timelineMeta.run_id) return timelineMeta;
  if (!AppState.selectedRunId) return null;
  return AppState.selectedOrgRuns.find(item => item.run_id === AppState.selectedRunId) || null;
}

function getRunStorageKey() {
  const organization = AppState.selectedOrganization || AppState.filters.organization || 'unknown-org';
  const runMeta = getCurrentRunMeta();
  const runId = AppState.selectedRunId || runMeta?.run_id || 'latest';
  return `${organization}::${runId}`;
}

function getExperimentRecordForCurrentRun() {
  return AppState.runNotes[getRunStorageKey()] || { tags: [], note: '', updatedAt: '' };
}

function getTeamAnnotationsForCurrentRun() {
  const rows = AppState.teamAnnotations[getRunStorageKey()];
  return Array.isArray(rows) ? rows : [];
}

function stableHash(value) {
  const text = String(value || '');
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash) + text.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

function buildShareStateParams() {
  const params = new URLSearchParams();
  params.set('page', AppState.currentPage || 'dashboard');

  const org = AppState.selectedOrganization || AppState.filters.organization;
  if (org) params.set('org', org);

  if (AppState.selectedRunId) params.set('run', AppState.selectedRunId);
  if (AppState.inputMode !== 'json') params.set('input', AppState.inputMode);

  if (AppState.filters.dateRange !== DEFAULT_VIEW_FILTERS.dateRange) {
    params.set('date', AppState.filters.dateRange);
  }
  if (AppState.filters.riskThreshold !== DEFAULT_VIEW_FILTERS.riskThreshold) {
    params.set('risk', AppState.filters.riskThreshold);
  }
  if (AppState.filters.sourceMode !== DEFAULT_VIEW_FILTERS.sourceMode) {
    params.set('source', AppState.filters.sourceMode);
  }

  if (AppState.comparisonSelection.length) {
    params.set('compare', AppState.comparisonSelection.slice(0, 6).join(','));
  }

  return params;
}

function buildShareableDashboardUrl() {
  const url = new URL(window.location.href);
  url.search = buildShareStateParams().toString();
  return url.toString();
}

function syncDeepLinkState() {
  if (AppState.deepLink.isApplying) return;
  const params = buildShareStateParams();
  const nextQuery = params.toString();
  const nextUrl = `${window.location.pathname}${nextQuery ? `?${nextQuery}` : ''}`;
  const current = `${window.location.pathname}${window.location.search}`;
  if (current !== nextUrl) {
    window.history.replaceState({}, '', nextUrl);
  }
}

function applyDeepLinkStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  if (!params.toString()) return false;

  AppState.deepLink.isApplying = true;

  const page = params.get('page');
  if (page && DASHBOARD_PAGES.includes(page)) {
    AppState.currentPage = page;
  }

  const inputMode = params.get('input');
  if (inputMode === 'json' || inputMode === 'query') {
    AppState.inputMode = inputMode;
  }

  const org = params.get('org');
  if (org) {
    AppState.selectedOrganization = org;
    AppState.filters.organization = org;
  }

  const runId = params.get('run');
  if (runId) {
    AppState.selectedRunId = runId;
    AppState.deepLink.pendingRunId = runId;
  }

  const dateRange = params.get('date');
  if (dateRange) AppState.filters.dateRange = dateRange;

  const riskThreshold = params.get('risk');
  if (riskThreshold) AppState.filters.riskThreshold = riskThreshold;

  const sourceMode = params.get('source');
  if (sourceMode) AppState.filters.sourceMode = sourceMode;

  const compare = params.get('compare');
  if (compare) {
    AppState.comparisonSelection = compare
      .split(',')
      .map(item => item.trim())
      .filter(Boolean)
      .slice(0, 6);
  }

  return true;
}

async function copyDashboardDeepLink() {
  const url = buildShareableDashboardUrl();
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(url);
    } else {
      const textarea = document.createElement('textarea');
      textarea.value = url;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }
    showToast('Shared dashboard link copied', 'success');
  } catch (err) {
    showToast(`Could not copy link: ${err.message}`, 'error');
  }
}

function computeSourceReliabilityScore(sourceBreakdown = {}) {
  const keys = ['analysis', 'risk', 'recommendation'];
  const samples = [];
  keys.forEach(key => {
    const source = String(sourceBreakdown?.[key] || '').trim().toLowerCase();
    if (!source) return;
    samples.push(SOURCE_RELIABILITY_SCORES[source] ?? 0.5);
  });

  if (!samples.length) return NaN;
  const average = samples.reduce((acc, value) => acc + value, 0) / samples.length;
  return average * 100;
}

function mean(values) {
  if (!Array.isArray(values) || !values.length) return NaN;
  const valid = values.filter(v => Number.isFinite(v));
  if (!valid.length) return NaN;
  return valid.reduce((acc, value) => acc + value, 0) / valid.length;
}

function stddev(values) {
  if (!Array.isArray(values) || values.length < 2) return NaN;
  const valid = values.filter(v => Number.isFinite(v));
  if (valid.length < 2) return NaN;
  const avg = mean(valid);
  const variance = valid.reduce((acc, value) => acc + ((value - avg) ** 2), 0) / valid.length;
  return Math.sqrt(variance);
}

function dateRangeDaysFromValue(value) {
  if (!value || value === 'all') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function getFilteredOrgRuns() {
  let runs = Array.isArray(AppState.selectedOrgRuns) ? [...AppState.selectedOrgRuns] : [];

  const days = dateRangeDaysFromValue(AppState.filters.dateRange);
  if (days !== null) {
    const cutoff = Date.now() - (days * 24 * 60 * 60 * 1000);
    runs = runs.filter(run => {
      const ts = new Date(run.created_at || '').getTime();
      return Number.isFinite(ts) && ts >= cutoff;
    });
  }

  if (AppState.filters.riskThreshold !== 'all') {
    const threshold = Number(AppState.filters.riskThreshold);
    if (Number.isFinite(threshold)) {
      runs = runs.filter(run => asNumber(run.summary?.risk_count, 0) >= threshold);
    }
  }

  if (AppState.filters.sourceMode !== 'all') {
    const selectedSource = String(AppState.filters.sourceMode || '').toLowerCase();
    runs = runs.filter(run => String(run.source || '').toLowerCase() === selectedSource);
  }

  return runs;
}

function syncTimelineFromFilters(orgChanged = false) {
  const filteredRuns = getFilteredOrgRuns();
  AppState.timeline.runs = [...filteredRuns].reverse();

  if (AppState.timeline.runs.length === 0) {
    AppState.timeline.currentIndex = 0;
    return filteredRuns;
  }

  if (orgChanged) {
    AppState.timeline.currentIndex = AppState.timeline.runs.length - 1;
  } else {
    AppState.timeline.currentIndex = Math.min(
      AppState.timeline.currentIndex,
      AppState.timeline.runs.length - 1,
    );
  }

  return filteredRuns;
}

async function applyViewFiltersFromControls() {
  const org = document.getElementById('view-filter-org')?.value || '';
  const dateRange = document.getElementById('view-filter-date')?.value || 'all';
  const riskThreshold = document.getElementById('view-filter-risk')?.value || 'all';
  const sourceMode = document.getElementById('view-filter-source')?.value || 'all';

  AppState.filters.organization = org;
  AppState.filters.dateRange = dateRange;
  AppState.filters.riskThreshold = riskThreshold;
  AppState.filters.sourceMode = sourceMode;

  persistActiveFilters();
  syncDeepLinkState();

  if (org && org !== AppState.selectedOrganization) {
    AppState.selectedOrganization = org;
    AppState.selectedRunId = '';
    await loadOrganizationHistory(org, { rerender: true });
    showToast('Applied saved-view filters', 'success');
    return;
  }

  syncTimelineFromFilters(false);
  try {
    await hydrateTimelineDetailsForCurrent();
  } catch (err) {
    console.warn('Could not hydrate timeline after filter apply', err);
  }
  derivePredictiveAlerts();
  renderDashboard();
  showToast('Applied saved-view filters', 'success');
}

function resetViewFilters() {
  AppState.filters = {
    ...DEFAULT_VIEW_FILTERS,
    organization: AppState.selectedOrganization || '',
  };
  persistActiveFilters();
  syncDeepLinkState();
  syncTimelineFromFilters(false);
  derivePredictiveAlerts();
  renderDashboard();
  showToast('Filters reset', 'info');
}

function saveCurrentFilterPreset() {
  const input = document.getElementById('preset-name-input');
  const presetName = input?.value.trim() || '';
  if (!presetName) {
    showToast('Provide a preset name first', 'info');
    return;
  }

  AppState.filterPresets[presetName] = {
    ...AppState.filters,
    organization: AppState.filters.organization || AppState.selectedOrganization || '',
    updatedAt: new Date().toISOString(),
  };

  persistFilterPresets();
  if (input) input.value = '';
  renderDashboard();
  showToast(`Saved preset: ${presetName}`, 'success');
}

async function applyFilterPreset(name) {
  const preset = AppState.filterPresets[name];
  if (!preset) {
    showToast('Preset not found', 'error');
    return;
  }

  AppState.filters = { ...DEFAULT_VIEW_FILTERS, ...normalizeFilterModel(preset) };
  persistActiveFilters();
  syncDeepLinkState();

  if (AppState.filters.organization && AppState.filters.organization !== AppState.selectedOrganization) {
    AppState.selectedOrganization = AppState.filters.organization;
    AppState.selectedRunId = '';
    await loadOrganizationHistory(AppState.selectedOrganization, { rerender: true });
  } else {
    syncTimelineFromFilters(false);
    try {
      await hydrateTimelineDetailsForCurrent();
    } catch (err) {
      console.warn('Could not hydrate timeline after applying preset', err);
    }
    derivePredictiveAlerts();
    renderDashboard();
  }

  showToast(`Applied preset: ${name}`, 'success');
}

function deleteFilterPreset(name) {
  if (!AppState.filterPresets[name]) return;
  delete AppState.filterPresets[name];
  persistFilterPresets();
  renderDashboard();
  showToast(`Deleted preset: ${name}`, 'info');
}

function buildShortcutActionsHtml() {
  return `
    <div class="btn-row" style="justify-content:center">
      <button class="btn btn-secondary" onclick="loadSampleShortcut()">Load Sample</button>
      <button class="btn btn-secondary" onclick="retrieveLatestShortcut()">Retrieve Latest</button>
      <button class="btn btn-primary" onclick="startNewRunShortcut()">Start New Run</button>
    </div>
  `;
}

function renderContextualEmptyState(page) {
  if (page === 'results') {
    const scoreGrid = document.getElementById('results-score-grid');
    if (scoreGrid) {
      scoreGrid.innerHTML = `
        <div class="card" style="grid-column:1/-1;text-align:center;padding:1.2rem">
          <div style="font-size:1.1rem;font-weight:700">No Results Yet</div>
          <div style="font-size:.74rem;color:var(--text-muted);margin:.3rem 0 .8rem">Run or retrieve a sprint to populate health scores and logs.</div>
          ${buildShortcutActionsHtml()}
        </div>
      `;
    }
  }

  if (page === 'dependencies') {
    const dep = document.getElementById('dep-graph-content');
    if (dep) {
      dep.innerHTML = `
        <div class="org-empty">
          <div style="font-size:.78rem;font-weight:600;margin-bottom:.25rem">No Dependency Graph Available</div>
          <div style="margin-bottom:.55rem">Run analysis or retrieve latest organization context to inspect dependency edges.</div>
          ${buildShortcutActionsHtml()}
        </div>
      `;
    }
  }

  if (page === 'recommendations') {
    const recs = document.getElementById('recs-list');
    if (recs) {
      recs.innerHTML = `
        <div class="card" style="text-align:center;padding:1.2rem">
          <div style="font-size:1.1rem;font-weight:700">No Recommendations Yet</div>
          <div style="font-size:.74rem;color:var(--text-muted);margin:.3rem 0 .8rem">Generate or load an inference run to view interventions and evidence.</div>
          ${buildShortcutActionsHtml()}
        </div>
      `;
    }
  }
}

async function loadSampleShortcut() {
  navigateTo('analyze');
  if (!AppState.sprintFiles.length) {
    await loadSprintFiles();
  }

  if (!AppState.sprintFiles.length) {
    loadExampleSprintInput();
    showToast('Dataset sample unavailable — loaded built-in example sprint', 'info');
    return;
  }

  await loadSprintFile(AppState.sprintFiles[0]);
}

async function retrieveLatestShortcut() {
  await refreshOrganizationIndex();
  if (!AppState.selectedOrganization) {
    showToast('No organization history available yet', 'info');
    return;
  }
  await retrieveOrganizationResult();
}

function startNewRunShortcut() {
  navigateTo('analyze');
  const runBtn = document.getElementById('btn-run');
  const sprintInput = document.getElementById('sprint-json');
  if (runBtn) runBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (sprintInput) sprintInput.focus();
}

function exportReportShortcut() {
  exportCurrentRun('json');
}

function normalizeTagList(rawTags) {
  const seen = new Set();
  const tags = [];
  rawTags.forEach(tag => {
    const normalized = String(tag || '').trim();
    if (!normalized) return;
    const key = normalized.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    tags.push(normalized);
  });
  return tags;
}

function buildReproducibilityMetadata() {
  const runMeta = getCurrentRunMeta() || {};
  const result = AppState.lastResult || {};
  const runMetrics = result.run_metrics || {};
  const evalMode = document.getElementById('eval-mode')?.value || AppState.settings.evalMode || 'resilient';
  const repositories = Array.isArray(runMeta.repositories) ? runMeta.repositories : [];

  const metadata = {
    generated_at: new Date().toISOString(),
    organization: AppState.selectedOrganization || null,
    run_id: AppState.selectedRunId || runMeta.run_id || null,
    run_created_at: runMeta.created_at || null,
    input_mode: AppState.inputMode,
    eval_mode: evalMode,
    source: runMeta.source || 'analyze',
    repositories,
    llm_backend: 'Ollama (Local)',
    vector_store: 'ChromaDB',
    analysis_source: result.analysis_source || runMetrics.source_breakdown?.analysis || null,
    risk_source: result.risk_source || runMetrics.source_breakdown?.risk || null,
    recommendation_source: result.recommendation_source || runMetrics.source_breakdown?.recommendation || null,
    app_version: '1.0.0-research-preview',
    state_url: buildShareableDashboardUrl(),
  };

  metadata.reproducibility_fingerprint = `rep-${stableHash(JSON.stringify([
    metadata.organization,
    metadata.run_id,
    metadata.eval_mode,
    metadata.source,
    metadata.repositories.join(','),
    metadata.analysis_source,
    metadata.risk_source,
    metadata.recommendation_source,
  ]))}`;

  return metadata;
}

function buildExportPayload() {
  const runKey = getRunStorageKey();
  return {
    metadata: buildReproducibilityMetadata(),
    filters: { ...AppState.filters },
    comparison_selection: [...AppState.comparisonSelection],
    experiment: AppState.runNotes[runKey] || { tags: [], note: '', updatedAt: '' },
    annotations: getTeamAnnotationsForCurrentRun(),
    result: AppState.lastResult,
  };
}

function formatRiskBand(score) {
  const value = asNumber(score, 0);
  if (value >= 0.7) return 'high';
  if (value >= 0.4) return 'medium';
  return 'low';
}

function buildMarkdownExport(payload) {
  const result = payload.result || {};
  const analysis = result.sprint_analysis || {};
  const risks = Array.isArray(result.identified_risks) ? result.identified_risks : [];
  const recs = Array.isArray(result.recommendations) ? result.recommendations : [];
  const citations = Array.isArray(result.evidence_citations) ? result.evidence_citations : [];

  const metricsRows = [
    ['Health Score', asNumber(analysis.health_score, 0).toFixed(1)],
    ['Health Status', analysis.health_status || 'unknown'],
    ['Completion Probability', `${asNumber(analysis.completion_probability, 0).toFixed(0)}%`],
    ['Delivery Score', asNumber(analysis.delivery_score, 0).toFixed(1)],
    ['Momentum Score', asNumber(analysis.momentum_score, 0).toFixed(1)],
    ['Quality Score', asNumber(analysis.quality_score, 0).toFixed(1)],
    ['Collaboration Score', asNumber(analysis.collaboration_score, 0).toFixed(1)],
    ['Dependency Risk Score', asNumber(analysis.dependency_risk_score, 0).toFixed(1)],
  ];

  const riskLines = risks.length
    ? risks.slice(0, 12).map(item => `- **${(item.risk_type || 'risk').replace(/_/g, ' ')}** (${formatRiskBand(item.severity)}): ${item.description || ''}`)
    : ['- No risks identified'];

  const recLines = recs.length
    ? recs.slice(0, 12).map(item => `- **${item.title || 'Recommendation'}** (${item.priority || 'medium'}): ${item.description || ''}`)
    : ['- No recommendations generated'];

  const citationLines = citations.length
    ? citations.map(item => `- ${item}`)
    : ['- No evidence citations'];

  const annotationRows = (payload.annotations || []).length
    ? payload.annotations.map(item => `- [${item.createdAt || 'n/a'}] (${item.decision || 'note'}) ${item.target || 'general'} :: ${item.comment || ''}`)
    : ['- No team annotations'];

  const tags = payload.experiment?.tags?.length ? payload.experiment.tags.join(', ') : 'none';
  const note = payload.experiment?.note || 'No analyst notes recorded.';

  return [
    '# Sprint Intelligence Export',
    '',
    '## Reproducibility Metadata',
    ...Object.entries(payload.metadata || {}).map(([key, value]) => `- ${key}: ${value == null ? '' : value}`),
    '',
    '## Sprint KPI Snapshot',
    '| Metric | Value |',
    '| --- | --- |',
    ...metricsRows.map(([label, value]) => `| ${label} | ${value} |`),
    '',
    '## Risks',
    ...riskLines,
    '',
    '## Recommendations',
    ...recLines,
    '',
    '## Evidence Citations',
    ...citationLines,
    '',
    '## Experiment Tags & Notes',
    `- tags: ${tags}`,
    `- note: ${note}`,
    '',
    '## Team Annotations',
    ...annotationRows,
    '',
  ].join('\n');
}

function downloadBlob(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function openPrintablePdf(markdown, metadata) {
  const printWindow = window.open('', '_blank', 'noopener,noreferrer,width=920,height=680');
  if (!printWindow) {
    showToast('Popup blocked. Allow popups to export PDF.', 'error');
    return;
  }

  const heading = `Sprint Intelligence Export - ${metadata.organization || 'organization'}`;
  const style = `
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; color: #0f172a; }
      h1 { font-size: 20px; margin-bottom: 8px; }
      .meta { color: #334155; font-size: 12px; margin-bottom: 16px; }
      pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.55; }
    </style>
  `;

  const html = `
    <html>
      <head><title>${escapeHtml(heading)}</title>${style}</head>
      <body>
        <h1>${escapeHtml(heading)}</h1>
        <div class="meta">Run ID: ${escapeHtml(metadata.run_id || 'n/a')} · Fingerprint: ${escapeHtml(metadata.reproducibility_fingerprint || '')}</div>
        <pre>${escapeHtml(markdown)}</pre>
      </body>
    </html>
  `;

  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
  setTimeout(() => {
    printWindow.focus();
    printWindow.print();
  }, 120);
}

function exportCurrentRun(format = 'json') {
  if (!AppState.lastResult) {
    showToast('No result available to export', 'info');
    return;
  }

  const payload = buildExportPayload();
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const org = payload.metadata.organization || 'org';
  const runId = payload.metadata.run_id || 'latest';
  const baseName = `${org}_${runId}_${stamp}`.replace(/[^A-Za-z0-9._-]+/g, '_');

  if (format === 'json') {
    downloadBlob(`sprint_export_${baseName}.json`, JSON.stringify(payload, null, 2), 'application/json');
    showToast('Exported report JSON', 'success');
    return;
  }

  const markdown = buildMarkdownExport(payload);
  if (format === 'md') {
    downloadBlob(`sprint_export_${baseName}.md`, markdown, 'text/markdown');
    showToast('Exported report Markdown', 'success');
    return;
  }

  if (format === 'pdf') {
    openPrintablePdf(markdown, payload.metadata || {});
    showToast('PDF export opened (print dialog)', 'success');
  }
}

function saveExperimentNotes() {
  if (!AppState.lastResult) {
    showToast('Load a run before adding experiment notes', 'info');
    return;
  }

  const tagsInput = document.getElementById('experiment-tags-input');
  const noteInput = document.getElementById('experiment-note-input');
  const rawTags = (tagsInput?.value || '').split(',');
  const tags = normalizeTagList(rawTags);
  const note = (noteInput?.value || '').trim();
  const storageKey = getRunStorageKey();

  AppState.runNotes[storageKey] = {
    tags,
    note,
    updatedAt: new Date().toISOString(),
  };

  persistRunNotes();
  renderDashboard();
  showToast('Saved experiment tags and notes', 'success');
}

function setDrilldownSearch(value) {
  AppState.drilldown.query = String(value || '');
  renderDashboard();
}

function setDrilldownSort(key) {
  const currentKey = AppState.drilldown.sortKey;
  const currentDir = AppState.drilldown.sortDir;
  if (currentKey === key) {
    AppState.drilldown.sortDir = currentDir === 'asc' ? 'desc' : 'asc';
  } else {
    AppState.drilldown.sortKey = key;
    AppState.drilldown.sortDir = key === 'metric' || key === 'category' ? 'asc' : 'desc';
  }
  renderDashboard();
}

function saveTeamAnnotation() {
  if (!AppState.lastResult) {
    showToast('Load a run before adding annotations', 'info');
    return;
  }

  const type = document.getElementById('annotation-type')?.value || 'general';
  const target = document.getElementById('annotation-target')?.value || 'general';
  const decision = document.getElementById('annotation-decision')?.value || 'monitor';
  const author = (document.getElementById('annotation-author')?.value || '').trim();
  const comment = (document.getElementById('annotation-comment')?.value || '').trim();

  if (!comment) {
    showToast('Add a comment before saving annotation', 'info');
    return;
  }

  const key = getRunStorageKey();
  const rows = Array.isArray(AppState.teamAnnotations[key]) ? AppState.teamAnnotations[key] : [];
  rows.unshift({
    id: `ann-${Date.now()}-${stableHash(comment).slice(0, 6)}`,
    type,
    target,
    decision,
    author: author || 'analyst',
    comment,
    createdAt: new Date().toISOString(),
  });

  AppState.teamAnnotations[key] = rows.slice(0, 120);
  persistTeamAnnotations();

  const commentInput = document.getElementById('annotation-comment');
  if (commentInput) commentInput.value = '';
  renderDashboard();
  showToast('Annotation saved to decision log', 'success');
}

function deleteTeamAnnotation(annotationId) {
  const key = getRunStorageKey();
  const rows = Array.isArray(AppState.teamAnnotations[key]) ? AppState.teamAnnotations[key] : [];
  const nextRows = rows.filter(item => item.id !== annotationId);
  AppState.teamAnnotations[key] = nextRows;
  persistTeamAnnotations();
  renderDashboard();
  showToast('Annotation removed', 'info');
}

function getCommandActions() {
  return [
    {
      id: 'cmd-run-analysis',
      title: 'Run Inference (Current Input)',
      meta: 'Analyze sprint using current input mode and form values.',
      keyHint: 'Alt+2',
      run: () => {
        navigateTo('analyze');
        runAnalysis('live');
      },
    },
    {
      id: 'cmd-load-latest-org',
      title: 'Load Latest Organization Result',
      meta: AppState.selectedOrganization
        ? `Current org: ${AppState.selectedOrganization}`
        : 'No organization selected yet.',
      keyHint: 'Org',
      disabled: !AppState.selectedOrganization,
      run: () => retrieveLatestShortcut(),
    },
    {
      id: 'cmd-open-dependencies',
      title: 'Open Dependencies Page',
      meta: 'Navigate to cross-repository dependency graph.',
      keyHint: 'Alt+4',
      run: () => navigateTo('dependencies'),
    },
    {
      id: 'cmd-open-about',
      title: 'Open About System Page',
      meta: 'View orchestration, pipeline, and inference architecture details.',
      keyHint: 'Alt+8',
      run: () => navigateTo('about'),
    },
    {
      id: 'cmd-export-report',
      title: 'Export Report JSON',
      meta: 'Download current sprint result, metadata, notes, and annotations.',
      keyHint: 'Export',
      disabled: !AppState.lastResult,
      run: () => exportCurrentRun('json'),
    },
    {
      id: 'cmd-export-markdown',
      title: 'Export Report Markdown',
      meta: 'Generate a reproducible markdown report.',
      keyHint: 'MD',
      disabled: !AppState.lastResult,
      run: () => exportCurrentRun('md'),
    },
    {
      id: 'cmd-export-pdf',
      title: 'Export Report PDF',
      meta: 'Open a print-ready report for PDF export.',
      keyHint: 'PDF',
      disabled: !AppState.lastResult,
      run: () => exportCurrentRun('pdf'),
    },
    {
      id: 'cmd-copy-link',
      title: 'Copy Shared Dashboard Link',
      meta: 'Copy deep-linked page/org/run/filter state.',
      keyHint: 'Link',
      run: () => copyDashboardDeepLink(),
    },
    {
      id: 'cmd-load-sample',
      title: 'Load Sample Sprint',
      meta: 'Load first available sprint JSON sample into editor.',
      keyHint: 'Sample',
      run: () => loadSampleShortcut(),
    },
  ];
}

function renderCommandPaletteList() {
  const listEl = document.getElementById('command-list');
  if (!listEl) return;

  const query = AppState.commandPalette.query.trim().toLowerCase();
  const commands = getCommandActions().filter(cmd => {
    if (!query) return true;
    return cmd.title.toLowerCase().includes(query) || cmd.meta.toLowerCase().includes(query);
  });

  AppState.commandPalette.commands = commands;

  if (!commands.length) {
    AppState.commandPalette.selectedIndex = 0;
    listEl.innerHTML = '<div class="command-empty">No matching commands found.</div>';
    return;
  }

  AppState.commandPalette.selectedIndex = Math.max(
    0,
    Math.min(AppState.commandPalette.selectedIndex, commands.length - 1),
  );

  listEl.innerHTML = commands.map((cmd, index) => {
    const isActive = index === AppState.commandPalette.selectedIndex;
    const disabledClass = cmd.disabled ? 'disabled' : '';
    return `<div class="command-item ${isActive ? 'active' : ''} ${disabledClass}" onclick="executeCommandFromPalette(${index})">
      <div>
        <div class="command-title">${escapeHtml(cmd.title)}</div>
        <div class="command-meta">${escapeHtml(cmd.meta)}</div>
      </div>
      <div class="command-key">${escapeHtml(cmd.keyHint || '')}</div>
    </div>`;
  }).join('');
}

function openCommandPalette() {
  const palette = document.getElementById('command-palette');
  const backdrop = document.getElementById('command-backdrop');
  const search = document.getElementById('command-search');
  if (!palette || !backdrop || !search) return;

  AppState.commandPalette.open = true;
  AppState.commandPalette.query = '';
  AppState.commandPalette.selectedIndex = 0;

  palette.classList.add('open');
  palette.setAttribute('aria-hidden', 'false');
  backdrop.classList.add('open');
  search.value = '';
  renderCommandPaletteList();

  setTimeout(() => search.focus(), 20);
}

function closeCommandPalette() {
  const palette = document.getElementById('command-palette');
  const backdrop = document.getElementById('command-backdrop');
  if (!palette || !backdrop) return;

  AppState.commandPalette.open = false;
  AppState.commandPalette.commands = [];
  palette.classList.remove('open');
  palette.setAttribute('aria-hidden', 'true');
  backdrop.classList.remove('open');
}

function onCommandSearchInput(value) {
  AppState.commandPalette.query = value || '';
  AppState.commandPalette.selectedIndex = 0;
  renderCommandPaletteList();
}

function executeCommandFromPalette(index) {
  const cmd = AppState.commandPalette.commands[index];
  if (!cmd) return;
  if (cmd.disabled) {
    showToast('This command is currently unavailable', 'info');
    return;
  }

  closeCommandPalette();
  setTimeout(() => cmd.run(), 0);
}

function onCommandSearchKeydown(event) {
  if (!AppState.commandPalette.open) return;

  if (event.key === 'ArrowDown') {
    event.preventDefault();
    AppState.commandPalette.selectedIndex = Math.min(
      AppState.commandPalette.selectedIndex + 1,
      Math.max(AppState.commandPalette.commands.length - 1, 0),
    );
    renderCommandPaletteList();
    return;
  }

  if (event.key === 'ArrowUp') {
    event.preventDefault();
    AppState.commandPalette.selectedIndex = Math.max(AppState.commandPalette.selectedIndex - 1, 0);
    renderCommandPaletteList();
    return;
  }

  if (event.key === 'Enter') {
    event.preventDefault();
    executeCommandFromPalette(AppState.commandPalette.selectedIndex);
    return;
  }

  if (event.key === 'Escape') {
    event.preventDefault();
    closeCommandPalette();
  }
}

function bindSidebarKeyboardNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.setAttribute('tabindex', '0');
    item.setAttribute('role', 'button');
    item.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        item.click();
      }
    });
  });
}

function isTypingTarget(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = (target.tagName || '').toLowerCase();
  return ['input', 'textarea', 'select', 'button'].includes(tag);
}

function getActivePageCards() {
  const page = document.querySelector('.page-view.active');
  if (!page) return [];

  return Array.from(page.querySelectorAll('.card, .stat-card, .item-card, .metric-tile, .compare-card'));
}

function refreshFocusableCards() {
  const cards = getActivePageCards();
  cards.forEach(card => {
    card.classList.add('focusable-card');
    card.setAttribute('tabindex', '-1');
  });

  if (!cards.length) {
    AppState.keyboard.cardFocusIndex = -1;
    return;
  }

  AppState.keyboard.cardFocusIndex = Math.min(
    Math.max(AppState.keyboard.cardFocusIndex, 0),
    cards.length - 1,
  );
}

function focusAdjacentCard(step) {
  const cards = getActivePageCards();
  if (!cards.length) {
    showToast('No focusable cards on this page', 'info');
    return;
  }

  if (AppState.keyboard.cardFocusIndex < 0) {
    AppState.keyboard.cardFocusIndex = step > 0 ? 0 : cards.length - 1;
  } else {
    AppState.keyboard.cardFocusIndex = (AppState.keyboard.cardFocusIndex + step + cards.length) % cards.length;
  }

  cards.forEach(card => card.setAttribute('tabindex', '-1'));
  const targetCard = cards[AppState.keyboard.cardFocusIndex];
  targetCard.setAttribute('tabindex', '0');
  targetCard.focus();
  targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function activateFocusedCardAction() {
  const cards = getActivePageCards();
  if (!cards.length || AppState.keyboard.cardFocusIndex < 0) return;

  const card = cards[AppState.keyboard.cardFocusIndex];
  const action = card.querySelector('button, a, [onclick]');
  if (action && typeof action.click === 'function') {
    action.click();
  } else {
    showToast('No direct action found on focused card', 'info');
  }
}

function handleGlobalKeyboardShortcuts(event) {
  const key = (event.key || '').toLowerCase();
  const comboCmdK = (event.metaKey || event.ctrlKey) && key === 'k';

  if (comboCmdK) {
    event.preventDefault();
    if (AppState.commandPalette.open) {
      closeCommandPalette();
    } else {
      openCommandPalette();
    }
    return;
  }

  if (event.key === 'Escape') {
    if (AppState.commandPalette.open) {
      closeCommandPalette();
      return;
    }

    closeExplainabilityDrawer();
    return;
  }

  if (AppState.commandPalette.open || isTypingTarget(event.target)) {
    return;
  }

  if (event.altKey && ['1', '2', '3', '4', '5', '6', '7', '8'].includes(event.key)) {
    event.preventDefault();
    const pageMap = {
      '1': 'dashboard',
      '2': 'analyze',
      '3': 'results',
      '4': 'dependencies',
      '5': 'recommendations',
      '6': 'ingestion',
      '7': 'settings',
      '8': 'about',
    };
    const page = pageMap[event.key];
    navigateTo(page);
    if (page === 'dashboard') {
      renderDashboard();
    }
    return;
  }

  if (event.altKey && event.key === '.') {
    event.preventDefault();
    focusAdjacentCard(1);
    return;
  }

  if (event.altKey && event.key === ',') {
    event.preventDefault();
    focusAdjacentCard(-1);
    return;
  }

  if (event.altKey && event.key === 'Enter') {
    event.preventDefault();
    activateFocusedCardAction();
  }
}

async function refreshOrganizationIndex() {
  try {
    const resp = await fetch('/api/results/orgs');
    if (!resp.ok) return;

    const data = await resp.json();
    AppState.organizationIndex = data.organizations || [];

    if (AppState.organizationIndex.length === 0) {
      AppState.selectedOrganization = '';
      AppState.selectedOrgRuns = [];
      AppState.filters.organization = '';
      AppState.comparisonSelection = [];
      AppState.comparisonResults = [];
      AppState.alerts = [];
      AppState.timeline.org = '';
      AppState.timeline.runs = [];
      AppState.timeline.currentIndex = 0;
      AppState.timeline.runDetailsCache = {};
      stopTimelinePlayback();
      persistActiveFilters();
      return;
    }

    const orgSet = new Set(AppState.organizationIndex.map(item => item.organization));
    AppState.comparisonSelection = AppState.comparisonSelection.filter(org => orgSet.has(org));
    if (AppState.comparisonSelection.length === 0) {
      AppState.comparisonSelection = AppState.organizationIndex
        .slice(0, Math.min(2, AppState.organizationIndex.length))
        .map(item => item.organization);
    }

    const filterOrgExists = AppState.filters.organization && orgSet.has(AppState.filters.organization);
    if (filterOrgExists) {
      AppState.selectedOrganization = AppState.filters.organization;
    } else if (!orgSet.has(AppState.selectedOrganization)) {
      AppState.selectedOrganization = AppState.organizationIndex[0].organization;
    }

    if (!AppState.filters.organization || !orgSet.has(AppState.filters.organization)) {
      AppState.filters.organization = AppState.selectedOrganization;
      persistActiveFilters();
    }

    await loadOrganizationHistory(AppState.selectedOrganization);

    if (AppState.deepLink.pendingRunId) {
      const pendingRunId = AppState.deepLink.pendingRunId;
      const exists = AppState.selectedOrgRuns.some(run => run.run_id === pendingRunId);
      if (exists) {
        await retrieveOrganizationResult(pendingRunId, { navigate: false, silent: true });
      }
      AppState.deepLink.pendingRunId = '';
      AppState.deepLink.isApplying = false;
      syncDeepLinkState();
    }
  } catch (err) {
    console.warn('Could not load organization index', err);
    AppState.deepLink.isApplying = false;
  }
}

async function loadOrganizationHistory(orgName, options = {}) {
  const { rerender = false } = options;
  if (!orgName) {
    AppState.selectedOrgRuns = [];
    AppState.filters.organization = '';
    AppState.timeline.org = '';
    AppState.timeline.runs = [];
    AppState.timeline.currentIndex = 0;
    AppState.timeline.runDetailsCache = {};
    stopTimelinePlayback();
    return;
  }

  const orgChanged = AppState.timeline.org !== orgName;

  try {
    const resp = await fetch(`/api/results/org/${encodeURIComponent(orgName)}/history?limit=40`);
    if (!resp.ok) {
      AppState.selectedOrgRuns = [];
      AppState.timeline.org = orgName;
      AppState.timeline.runs = [];
      AppState.timeline.currentIndex = 0;
      if (orgChanged) {
        AppState.timeline.runDetailsCache = {};
      }
      stopTimelinePlayback();
      AppState.alerts = [];
      return;
    }

    const data = await resp.json();
    AppState.selectedOrgRuns = data.runs || [];

    if (orgChanged) {
      AppState.timeline.runDetailsCache = {};
      stopTimelinePlayback();
    }

    AppState.timeline.org = orgName;
    AppState.filters.organization = orgName;
    persistActiveFilters();

    const filteredRuns = syncTimelineFromFilters(orgChanged);

    if (AppState.selectedRunId) {
      const selectedIndex = AppState.timeline.runs.findIndex(run => run.run_id === AppState.selectedRunId);
      if (selectedIndex >= 0) {
        AppState.timeline.currentIndex = selectedIndex;
      }
    }

    if (!filteredRuns.length) {
      AppState.alerts = [];
    }

    await hydrateTimelineDetailsWindow(10);
    await hydrateTimelineDetailsForCurrent();
    derivePredictiveAlerts();
    syncDeepLinkState();

    if (rerender && AppState.currentPage === 'dashboard') {
      renderDashboard();
    }
  } catch (err) {
    console.warn('Could not load org history', err);
    AppState.selectedOrgRuns = [];
    AppState.alerts = [];
  }
}

async function onOrganizationChange(orgName) {
  AppState.selectedOrganization = orgName;
  AppState.selectedRunId = '';
  AppState.filters.organization = orgName;
  persistActiveFilters();
  syncDeepLinkState();
  await loadOrganizationHistory(orgName, { rerender: true });
}

async function retrieveOrganizationResult(runId = '', options = {}) {
  const { navigate = true, silent = false } = options;
  const selectedOrg = AppState.selectedOrganization || document.getElementById('org-select')?.value || '';
  if (!selectedOrg) {
    showToast('Select an organization first', 'info');
    return;
  }

  const runIdQuery = runId ? `?run_id=${encodeURIComponent(runId)}` : '';

  try {
    const resp = await fetch(`/api/results/org/${encodeURIComponent(selectedOrg)}${runIdQuery}`);
    const data = await resp.json();

    if (!resp.ok || data.error) {
      throw new Error(data.error || 'Could not retrieve organization result');
    }

    AppState.lastResult = data.result || null;
    AppState.selectedOrganization = data.organization || selectedOrg;
    AppState.filters.organization = AppState.selectedOrganization;
    AppState.selectedRunId = data.entry?.run_id || '';
    syncDeepLinkState();

    if (!AppState.lastResult) {
      throw new Error('No result payload found for selected organization');
    }

    renderResults(AppState.lastResult);
    await loadOrganizationHistory(AppState.selectedOrganization);
    derivePredictiveAlerts();
    renderDashboard();
    if (navigate) {
      navigateTo('dashboard');
    }
    if (!silent) {
      showToast(`Loaded ${AppState.selectedOrganization} result`, 'success');
    }
  } catch (err) {
    if (!silent) {
      showToast(`Retrieve failed: ${err.message}`, 'error');
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Sprint data file loading
// ═══════════════════════════════════════════════════════════════

async function loadSprintFiles() {
  try {
    const resp = await fetch('/api/data/sprints');
    if (!resp.ok) return;
    const data = await resp.json();
    AppState.sprintFiles = data.files || [];
    renderFileChips();
  } catch (_) { /* ignore */ }
}

function renderFileChips() {
  const container = document.getElementById('file-chips');
  if (!container) return;
  container.innerHTML = AppState.sprintFiles.map(f =>
    `<div class="file-chip" onclick="loadSprintFile('${f}', this)" title="Load ${f}">${f.replace('data/', '').replace('.json', '')}</div>`
  ).join('');
}

async function loadSprintFile(filename, chipElement) {
  try {
    const resp = await fetch(`/api/data/sprint/${encodeURIComponent(filename)}`);
    if (!resp.ok) throw new Error('Failed to load');
    const data = await resp.json();

    // Auto-fill owner/repo from first sprint
    const sprint = Array.isArray(data) ? data[0] : data;
    if (sprint && sprint.repo) {
      const parts = sprint.repo.split('/');
      if (parts.length === 2) {
        document.getElementById('owner-input').value = parts[0];
        document.getElementById('repo-input').value = parts[1];
      }
    }

    // Put JSON into editor (first sprint only for readability, or full array)
    const jsonStr = JSON.stringify(Array.isArray(data) ? data[0] : data, null, 2);
    document.getElementById('sprint-json').value = jsonStr;

    // Mark active chip
    document.querySelectorAll('.file-chip').forEach(el => el.classList.remove('active'));
    if (chipElement) {
      chipElement.classList.add('active');
    }
    showToast(`Loaded: ${filename}`, 'success');
  } catch (err) {
    showToast(`Failed to load: ${err.message}`, 'error');
  }
}

function handleFileUpload(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('sprint-json').value = e.target.result;
    showToast(`Uploaded: ${file.name}`, 'success');

    // Try to parse and auto-fill owner/repo
    try {
      const data = JSON.parse(e.target.result);
      const sprint = Array.isArray(data) ? data[0] : data;
      if (sprint && sprint.repo) {
        const parts = sprint.repo.split('/');
        if (parts.length === 2) {
          document.getElementById('owner-input').value = parts[0];
          document.getElementById('repo-input').value = parts[1];
        }
      }
    } catch (_) { /* ignore parse errors */ }
  };
  reader.readAsText(file);
}

function formatJson() {
  const textarea = document.getElementById('sprint-json');
  try {
    const parsed = JSON.parse(textarea.value);
    textarea.value = JSON.stringify(parsed, null, 2);
    showToast('JSON formatted', 'success');
  } catch (e) {
    showToast('Invalid JSON: ' + e.message, 'error');
  }
}

function clearJson() {
  document.getElementById('sprint-json').value = '';
  document.querySelectorAll('.file-chip').forEach(el => el.classList.remove('active'));
}

function loadExampleSprintInput() {
  setInputMode('json');

  const sprintInput = document.getElementById('sprint-json');
  const ownerInput = document.getElementById('owner-input');
  const repoInput = document.getElementById('repo-input');

  if (sprintInput) {
    sprintInput.value = JSON.stringify(EXAMPLE_SPRINT_INPUT, null, 2);
    sprintInput.focus();
  }

  if (ownerInput && repoInput) {
    const [owner, repo] = EXAMPLE_SPRINT_INPUT.repo.split('/');
    ownerInput.value = owner || '';
    repoInput.value = repo || '';
  }

  document.querySelectorAll('.file-chip').forEach(el => el.classList.remove('active'));
  showToast('Loaded built-in example sprint input', 'success');
}

// ═══════════════════════════════════════════════════════════════
// Pipeline visualization
// ═══════════════════════════════════════════════════════════════

function resetPipeline() {
  document.querySelectorAll('.agent-node').forEach(n => {
    n.classList.remove('active', 'done', 'error');
  });
}

function advancePipeline(idx) {
  const nodes = document.querySelectorAll('.agent-node');
  nodes.forEach((n, i) => {
    if (i < idx) { n.classList.add('done'); n.classList.remove('active'); }
    else if (i === idx) { n.classList.add('active'); n.classList.remove('done'); }
    else { n.classList.remove('active', 'done'); }
  });
}

// ═══════════════════════════════════════════════════════════════
// Analysis execution
// ═══════════════════════════════════════════════════════════════

async function runAnalysis(mode) {
  if (AppState.isRunning) return;
  AppState.isRunning = true;

  const btnRun = document.getElementById('btn-run');
  const btnMock = document.getElementById('btn-mock');
  const statusBar = document.getElementById('status-bar');
  const statusText = document.getElementById('status-text');
  const consoleLogs = document.getElementById('console-logs');
  const errorBanner = document.getElementById('error-banner');
  const resultsArea = document.getElementById('results-area');

  btnRun.disabled = btnMock.disabled = true;
  errorBanner.classList.remove('visible');
  resultsArea.classList.remove('visible');
  consoleLogs.innerHTML = '';
  consoleLogs.classList.add('visible');
  statusBar.classList.add('visible');
  resetPipeline();
  stageIdx = 0;

  // Pipeline animation
  animTimer = setInterval(() => {
    if (stageIdx < STAGES.length) {
      advancePipeline(stageIdx);
      statusText.textContent = `[${stageIdx + 1}/${STAGES.length}] ${STAGES[stageIdx].name} — ${STAGES[stageIdx].desc}`;
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = `[${new Date().toLocaleTimeString()}] Agent ${stageIdx + 1}: ${STAGES[stageIdx].name} → ${STAGES[stageIdx].desc}`;
      consoleLogs.appendChild(line);
      consoleLogs.scrollTop = consoleLogs.scrollHeight;
      stageIdx++;
    }
  }, 1600);

  // Build payload
  const owner = document.getElementById('owner-input').value.trim();
  const repo = document.getElementById('repo-input').value.trim();
  const evalMode = document.getElementById('eval-mode').value;
  const sprintJsonStr = document.getElementById('sprint-json').value.trim();
  const queryText = document.getElementById('query-input')?.value.trim() || '';

  let url, payload;

  if (mode === 'mock') {
    url = '/api/analyze/mock';
    payload = { eval_mode: evalMode };
  } else if (AppState.inputMode === 'query') {
    if (!owner || !repo) {
      clearInterval(animTimer);
      statusText.textContent = '❌ Missing repository context';
      document.getElementById('error-text').textContent = 'Please provide both owner and repository for query input mode.';
      errorBanner.classList.add('visible');
      btnRun.disabled = btnMock.disabled = false;
      AppState.isRunning = false;
      return;
    }

    if (!queryText) {
      clearInterval(animTimer);
      statusText.textContent = '❌ Missing query text';
      document.getElementById('error-text').textContent = 'Please enter query input text before running inference.';
      errorBanner.classList.add('visible');
      btnRun.disabled = btnMock.disabled = false;
      AppState.isRunning = false;
      return;
    }

    url = '/api/analyze/query';
    payload = {
      owner,
      repo,
      query_text: queryText,
      eval_mode: evalMode,
      model: AppState.models.selected || undefined,
    };
  } else if (sprintJsonStr) {
    url = '/api/analyze/sprint';
    let sprintData;
    try {
      sprintData = JSON.parse(sprintJsonStr);
    } catch (e) {
      clearInterval(animTimer);
      statusText.textContent = '❌ Invalid JSON input';
      document.getElementById('error-text').textContent = 'Sprint JSON is not valid: ' + e.message;
      errorBanner.classList.add('visible');
      btnRun.disabled = btnMock.disabled = false;
      AppState.isRunning = false;
      return;
    }
    payload = {
      owner: owner,
      repo: repo,
      sprint_data: sprintData,
      eval_mode: evalMode,
      model: AppState.models.selected || undefined,
    };
  } else {
    url = '/api/analyze';
    const repoStr = owner && repo ? `${owner}/${repo}` : repo || owner;
    if (!repoStr) {
      clearInterval(animTimer);
      statusText.textContent = '❌ Missing repository';
      document.getElementById('error-text').textContent = 'Please provide owner and repository name, or paste sprint JSON data.';
      errorBanner.classList.add('visible');
      btnRun.disabled = btnMock.disabled = false;
      AppState.isRunning = false;
      return;
    }
    payload = { repositories: [repoStr], eval_mode: evalMode, model: AppState.models.selected || undefined };
  }

  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    clearInterval(animTimer);
    const data = await resp.json();

    if (!resp.ok || data.error) {
      throw { message: data.error || 'Unknown error', traceback: data.traceback || '' };
    }

    // Finish remaining stages
    while (stageIdx < STAGES.length) {
      const line = document.createElement('div');
      line.className = 'log-line';
      line.textContent = `[${new Date().toLocaleTimeString()}] Agent ${stageIdx + 1}: ${STAGES[stageIdx].name} ✓`;
      consoleLogs.appendChild(line);
      stageIdx++;
    }
    advancePipeline(STAGES.length);
    document.querySelectorAll('.agent-node').forEach(n => n.classList.add('done'));
    statusText.textContent = '✅ Pipeline complete — all 11 agents finished';

    AppState.lastResult = data.result;
    AppState.selectedOrganization = data.organization || AppState.selectedOrganization;
    AppState.selectedRunId = data.run_id || '';
    syncDeepLinkState();
    await refreshOrganizationIndex();
    if (AppState.selectedOrganization) {
      await loadOrganizationHistory(AppState.selectedOrganization);
    }

    renderResults(data.result);
    showToast('Analysis complete!', 'success');

    // Update nav badge
    document.querySelector('[data-page="results"]').classList.add('has-data');

  } catch (e) {
    clearInterval(animTimer);
    statusText.textContent = '❌ Pipeline failed';
    document.getElementById('error-text').textContent =
      (e.message || e) + (e.traceback ? '\n\n' + e.traceback : '');
    errorBanner.classList.add('visible');
    showToast('Analysis failed', 'error');
  } finally {
    btnRun.disabled = btnMock.disabled = false;
    AppState.isRunning = false;
  }
}

// ═══════════════════════════════════════════════════════════════
// Results rendering
// ═══════════════════════════════════════════════════════════════

function colorClass(v) { return v >= 70 ? 'green' : v >= 45 ? 'amber' : 'red'; }
function statusLabel(s) {
  return { on_track: 'On Track', at_risk: 'At Risk', critical: 'Critical' }[s] || s || 'Unknown';
}
function sourceBadge(s) {
  if (!s) return '';
  const cls = s === 'llm' ? 'badge-llm' : s === 'fallback' ? 'badge-fallback' : 'badge-low';
  return `<span class="badge ${cls}">source: ${s}</span>`;
}

function buildRecommendationsHtml(recs, emptyMessage) {
  return recs.length ? recs.map(rec => {
    const priority = rec.priority || 'medium';
    return `<div class="item-card">
      <div class="item-header"><span class="item-title">${rec.title || 'Recommendation'}</span><span class="badge badge-${priority}">${priority}</span></div>
      <div class="item-desc">${rec.description || ''}</div>
      ${rec.action ? `<div class="item-desc" style="margin-top:.2rem;color:var(--emerald)">→ ${rec.action}</div>` : ''}
      ${rec.expected_impact ? `<div class="item-desc" style="margin-top:.15rem;color:var(--text-muted)">Impact: ${rec.expected_impact}</div>` : ''}
      ${rec.evidence_source ? `<div class="item-desc" style="margin-top:.1rem;color:var(--cyan);font-size:.68rem">Evidence: ${rec.evidence_source}</div>` : ''}
    </div>`;
  }).join('') : `<div class="item-card"><div class="item-desc">${emptyMessage}</div></div>`;
}

function buildRisksHtml(risks, emptyMessage) {
  return risks.length ? risks.map(risk => {
    const sev = risk.severity ?? 0;
    const band = sev >= 0.7 ? 'high' : sev >= 0.4 ? 'medium' : 'low';
    return `<div class="item-card"><div class="item-header"><span class="item-title">${(risk.risk_type || 'Risk').replace(/_/g, ' ')}</span><span class="badge badge-${band}">${band} · ${(sev * 100).toFixed(0)}%</span></div><div class="item-desc">${risk.description || ''}</div>${risk.affected_issues?.length ? `<div class="item-desc" style="margin-top:.2rem;color:var(--amber)">Affected issues: ${risk.affected_issues.join(', ')}</div>` : ''}</div>`;
  }).join('') : `<div class="item-card"><div class="item-desc">${emptyMessage}</div></div>`;
}

function buildRagHtml(r) {
  const sims = r.similar_sprint_ids || [];
  const cites = r.evidence_citations || [];
  const synth = r.synthetic_sprints || [];
  const synthVal = r.synthetic_validation || {};

  let ragHTML = `<div style="font-size:.72rem;margin-bottom:.4rem"><strong style="color:var(--cyan)">Similar Historical Sprints:</strong> ${sims.length ? sims.join(', ') : 'None retrieved'}</div>`;
  ragHTML += `<div style="font-size:.72rem;margin-bottom:.4rem"><strong style="color:var(--cyan)">Synthetic Sprints:</strong> ${synth.length} scenarios · Embedded: ${r.synthetic_embedded_count || 0}</div>`;
  if (Object.keys(synthVal).length) {
    ragHTML += `<div style="font-size:.72rem;margin-bottom:.4rem"><strong style="color:var(--cyan)">Synthetic Validation:</strong> Realism score: ${(synthVal.realism_score ?? 0).toFixed(3)}</div>`;
  }
  const citationHtml = cites.length
    ? `<div style="margin-top:.2rem">${cites.slice(0, 20).map(url => `<div style="margin:.12rem 0;word-break:break-all">• ${escapeHtml(url)}</div>`).join('')}</div>`
    : '<span style="color:var(--text-muted)">No citations</span>';
  ragHTML += `<div style="font-size:.72rem"><strong style="color:var(--cyan)">Evidence Citations:</strong>${citationHtml}</div>`;
  return ragHTML;
}

function buildDependenciesHtml(r) {
  const dep = r.dependency_graph || {};
  const nodes = dep.nodes || [];
  const edges = dep.edges || [];
  const propagation = dep.risk_propagation || {};

  let html = `<div style="margin-bottom:.5rem;font-size:.72rem;color:var(--text-muted)">Nodes: ${nodes.length} repositories · Edges: ${edges.length} dependencies</div>`;

  if (nodes.length) {
    html += '<div style="margin-bottom:.75rem">' + nodes.map(n => {
      const riskP = propagation[n];
      const riskStr = riskP !== undefined ? ` (propagation: ${(riskP * 100).toFixed(0)}%)` : '';
      return `<span class="dep-node">${n}${riskStr}</span>`;
    }).join(' ') + '</div>';
  }

  if (edges.length) {
    html += edges.map(e =>
      `<div class="dep-edge"><span>${e.source}</span><span class="arrow">→</span><span>${e.target}</span><span style="color:var(--text-muted)">[${e.type}]</span>${e.is_blocker ? '<span class="badge badge-high">blocker</span>' : ''}</div>`
    ).join('');
  } else {
    html += '<div style="font-size:.72rem;color:var(--text-muted)">No cross-repo dependencies detected (single-repo analysis)</div>';
  }

  return html;
}

function renderResults(r) {
  const resultsArea = document.getElementById('results-area');

  const a = r.sprint_analysis || {};
  const health = a.health_score ?? 0;
  const comp = a.completion_probability ?? 0;
  const deliv = a.delivery_score ?? 0;
  const mom = a.momentum_score ?? 0;
  const qual = a.quality_score ?? 0;
  const collab = a.collaboration_score ?? 0;
  const depRisk = a.dependency_risk_score ?? 0;

  // Score grid
  document.getElementById('score-grid').innerHTML = `
    <div class="stat-card ${colorClass(health)}"><div class="stat-label">Health Score</div><div class="stat-value">${health.toFixed(1)}</div><div class="stat-sub">${statusLabel(a.health_status)}</div></div>
    <div class="stat-card ${colorClass(comp)}"><div class="stat-label">Completion</div><div class="stat-value">${comp.toFixed(0)}%</div><div class="stat-sub">Predicted probability</div></div>
    <div class="stat-card ${colorClass(deliv)}"><div class="stat-label">Delivery</div><div class="stat-value">${deliv.toFixed(1)}</div><div class="stat-sub">Issue + PR rate</div></div>
    <div class="stat-card ${colorClass(mom)}"><div class="stat-label">Momentum</div><div class="stat-value">${mom.toFixed(1)}</div><div class="stat-sub">Commit frequency</div></div>
    <div class="stat-card ${colorClass(qual)}"><div class="stat-label">Quality</div><div class="stat-value">${qual.toFixed(1)}</div><div class="stat-sub">Code concentration</div></div>
    <div class="stat-card ${colorClass(collab)}"><div class="stat-label">Collaboration</div><div class="stat-value">${collab.toFixed(1)}</div><div class="stat-sub">Author participation</div></div>
    <div class="stat-card ${colorClass(100 - depRisk)}"><div class="stat-label">Dep Risk</div><div class="stat-value">${depRisk.toFixed(1)}</div><div class="stat-sub">Cross-repo propagation</div></div>
  `;

  // Research metrics
  const rm = r.run_metrics || {};
  const cq = rm.citation_quality || {};
  const counts = rm.counts || {};
  document.getElementById('run-metrics').innerHTML = `
    <div class="metric-tile"><div class="m-label">Latency</div><div class="m-val">${(rm.latency_seconds ?? 0).toFixed(2)}s</div><div class="m-sub">End-to-end pipeline</div></div>
    <div class="metric-tile"><div class="m-label">F1 Score</div><div class="m-val">${rm.f1_score != null ? rm.f1_score.toFixed(3) : '—'}</div><div class="m-sub">Target ≥0.85</div></div>
    <div class="metric-tile"><div class="m-label">Parse Success</div><div class="m-val">${rm.parse_success_rate != null ? (rm.parse_success_rate * 100).toFixed(0) + '%' : '—'}</div><div class="m-sub">LLM output quality</div></div>
    <div class="metric-tile"><div class="m-label">Fallback Rate</div><div class="m-val">${rm.fallback_rate != null ? (rm.fallback_rate * 100).toFixed(0) + '%' : '—'}</div><div class="m-sub">Deterministic fallback</div></div>
    <div class="metric-tile"><div class="m-label">Citation Quality</div><div class="m-val">${(cq.score ?? 0).toFixed(2)}</div><div class="m-sub">${cq.non_empty_citations ?? 0}/${cq.total_citations ?? 0} citations</div></div>
    <div class="metric-tile"><div class="m-label">Analysis Source</div><div class="m-val" style="font-size:.85rem">${(rm.source_breakdown || {}).analysis || '—'}</div><div class="m-sub">LLM / fallback / error</div></div>
    <div class="metric-tile"><div class="m-label">Risk Source</div><div class="m-val" style="font-size:.85rem">${(rm.source_breakdown || {}).risk || '—'}</div><div class="m-sub">${counts.risks ?? 0} risks detected</div></div>
    <div class="metric-tile"><div class="m-label">Rec Source</div><div class="m-val" style="font-size:.85rem">${(rm.source_breakdown || {}).recommendation || '—'}</div><div class="m-sub">${counts.recommendations ?? 0} recommendations</div></div>
  `;

  // Features breakdown
  renderFeatures(r.features || {});

  // Charts
  renderCharts(r);

  // Recommendations on analysis page
  const recs = Array.isArray(r.recommendations) ? r.recommendations : [];
  const analyzeRecSource = document.getElementById('analyze-recs-source');
  const analyzeRecList = document.getElementById('analyze-recs-list');
  if (analyzeRecSource) analyzeRecSource.innerHTML = sourceBadge(r.recommendation_source);
  if (analyzeRecList) {
    analyzeRecList.innerHTML = buildRecommendationsHtml(
      recs,
      'No recommendations generated yet. Run an analysis first.'
    );
  }

  // Risks, evidence, and narrative on analysis page
  const risks = Array.isArray(r.identified_risks) ? r.identified_risks : [];
  const analyzeRiskSource = document.getElementById('analyze-risk-source');
  const analyzeRiskList = document.getElementById('analyze-risks');
  const analyzeDepGraph = document.getElementById('analyze-dep-graph');
  const analyzeRag = document.getElementById('analyze-rag');
  const analyzeNarrative = document.getElementById('analyze-narrative');
  if (analyzeRiskSource) analyzeRiskSource.innerHTML = sourceBadge(r.risk_source);
  if (analyzeRiskList) {
    analyzeRiskList.innerHTML = buildRisksHtml(
      risks,
      'No risks identified in this analysis.'
    );
  }
  if (analyzeDepGraph) analyzeDepGraph.innerHTML = buildDependenciesHtml(r);
  if (analyzeRag) analyzeRag.innerHTML = buildRagHtml(r);
  if (analyzeNarrative) {
    const analyzeNarrativeText = r.narrative_explanation || 'No narrative generated.';
    analyzeNarrative.innerHTML = renderNarrativeMarkdown(analyzeNarrativeText);
  }

  // Show results on analyze page
  resultsArea.classList.add('visible');

  // Also populate results page
  renderResultsPage(r);
  renderDependenciesPage(r);
  renderRecommendationsPage(r);

  const eLogs = r.execution_logs || [];
  const errs = r.errors || [];
  const analyzeExecLogs = document.getElementById('analyze-exec-logs');
  if (analyzeExecLogs) {
    analyzeExecLogs.innerHTML =
      eLogs.map(l => `<div>${l}</div>`).join('') +
      (errs.length ? '<div style="color:var(--rose);margin-top:.3rem">-- Errors --</div>' : '') +
      errs.map(e => `<div style="color:#fda4af">! ${e}</div>`).join('');
  }

  renderDashboard();
  refreshFocusableCards();
}

function renderFeatures(feats) {
  const catIcons = { temporal: '🕐', activity: '📊', code: '💻', risk: '⚠️', team: '👥', language: '🔤' };
  let html = '';
  for (const [cat, metrics] of Object.entries(feats)) {
    if (!metrics || typeof metrics !== 'object') continue;
    let rows = '';
    for (const [k, v] of Object.entries(metrics)) {
      const display = typeof v === 'number' ? v.toFixed(3) : String(v);
      rows += `<div class="feat-row"><span class="fname">${k.replace(/_/g, ' ')}</span><span class="fval">${display}</span></div>`;
    }
    if (rows) html += `<div class="feat-cat"><div class="feat-cat-title">${catIcons[cat] || '📐'} ${cat.toUpperCase()}</div>${rows}</div>`;
  }
  document.getElementById('features-panel').innerHTML = html || '<div style="color:var(--text-muted);font-size:.78rem">No features extracted</div>';
}

// ═══════════════════════════════════════════════════════════════
// Charts (Chart.js)
// ═══════════════════════════════════════════════════════════════

function renderCharts(r) {
  const a = r.sprint_analysis || {};

  // Destroy existing charts
  Object.values(AppState.charts).forEach(c => c.destroy && c.destroy());
  AppState.charts = {};

  // Health radar chart
  const radarCtx = document.getElementById('chart-radar');
  if (radarCtx) {
    AppState.charts.radar = new Chart(radarCtx, {
      type: 'radar',
      data: {
        labels: ['Health', 'Delivery', 'Momentum', 'Quality', 'Collaboration', 'Completion'],
        datasets: [{
          label: 'Sprint Scores',
          data: [
            a.health_score ?? 0,
            a.delivery_score ?? 0,
            a.momentum_score ?? 0,
            a.quality_score ?? 0,
            a.collaboration_score ?? 0,
            a.completion_probability ?? 0,
          ],
          borderColor: '#818cf8',
          backgroundColor: '#818cf820',
          borderWidth: 2,
          pointBackgroundColor: '#818cf8',
          pointBorderColor: '#fff',
          pointBorderWidth: 1,
          pointRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: { color: '#64748b', backdropColor: 'transparent', font: { size: 10 } },
            grid: { color: '#1a254040' },
            angleLines: { color: '#1a254040' },
            pointLabels: { color: '#94a3b8', font: { size: 11, weight: '600' } },
          },
        },
      },
    });
  }

  // Feature breakdown bar chart
  const barCtx = document.getElementById('chart-features');
  if (barCtx && r.features) {
    const featureLabels = [];
    const featureValues = [];
    const featureColors = [];
    const palette = ['#818cf8', '#22d3ee', '#34d399', '#fbbf24', '#fb7185', '#c084fc'];
    let ci = 0;
    for (const [cat, metrics] of Object.entries(r.features)) {
      if (!metrics || typeof metrics !== 'object') continue;
      const color = palette[ci % palette.length];
      for (const [k, v] of Object.entries(metrics)) {
        if (typeof v === 'number') {
          featureLabels.push(k.replace(/_/g, ' '));
          featureValues.push(v);
          featureColors.push(color);
        }
      }
      ci++;
    }

    AppState.charts.features = new Chart(barCtx, {
      type: 'bar',
      data: {
        labels: featureLabels.slice(0, 12),
        datasets: [{
          data: featureValues.slice(0, 12),
          backgroundColor: featureColors.slice(0, 12).map(c => c + '60'),
          borderColor: featureColors.slice(0, 12),
          borderWidth: 1,
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: '#1a254020' },
          },
          y: {
            ticks: { color: '#94a3b8', font: { size: 10 } },
            grid: { display: false },
          },
        },
      },
    });
  }

  // Risk severity doughnut
  const doughnutCtx = document.getElementById('chart-risks');
  if (doughnutCtx) {
    const risks = r.identified_risks || [];
    const high = risks.filter(r => (r.severity ?? 0) >= 0.7).length;
    const med = risks.filter(r => (r.severity ?? 0) >= 0.4 && (r.severity ?? 0) < 0.7).length;
    const low = risks.filter(r => (r.severity ?? 0) < 0.4).length;

    AppState.charts.risks = new Chart(doughnutCtx, {
      type: 'doughnut',
      data: {
        labels: ['High', 'Medium', 'Low'],
        datasets: [{
          data: [high, med, low],
          backgroundColor: ['#fb718560', '#fbbf2460', '#34d39960'],
          borderColor: ['#fb7185', '#fbbf24', '#34d399'],
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { size: 11 }, padding: 16 },
          },
        },
      },
    });
  }
}

// ═══════════════════════════════════════════════════════════════
// Results page rendering
// ═══════════════════════════════════════════════════════════════

function renderNarrativeMarkdown(rawText) {
  const safe = escapeHtml(rawText || '');
  return safe
    .replace(/^###\s+(.+)$/gm, '<div style="color:var(--indigo);font-weight:700;font-size:.92rem;margin:.55rem 0 .2rem">$1</div>')
    .replace(/^##\s+(.+)$/gm, '<div style="color:var(--indigo);font-weight:800;font-size:1rem;margin:.7rem 0 .25rem">$1</div>')
    .replace(/^#\s+(.+)$/gm, '<div style="color:var(--text);font-weight:800;font-size:1.08rem;margin:.2rem 0 .35rem">$1</div>')
    .replace(/^\d+\.\s+(.+)$/gm, '<div style="margin-left:.2rem;margin-top:.22rem">$&</div>')
    .replace(/^-\s+(.+)$/gm, '<div style="margin-left:.2rem;margin-top:.18rem">• $1</div>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function renderResultsPage(r) {
  const a = r.sprint_analysis || {};
  const rm = r.run_metrics || {};

  // Score summary on results page
  const el = document.getElementById('results-score-grid');
  if (el) {
    el.innerHTML = document.getElementById('score-grid').innerHTML;
  }

  // Risks
  document.getElementById('results-risk-source').innerHTML = sourceBadge(r.risk_source);
  const risks = Array.isArray(r.identified_risks) ? r.identified_risks : [];
  document.getElementById('results-risks').innerHTML = buildRisksHtml(
    risks,
    'No risks identified.'
  );

  // Recommendations
  const recs = Array.isArray(r.recommendations) ? r.recommendations : [];
  const resultsRecSource = document.getElementById('results-rec-source');
  if (resultsRecSource) resultsRecSource.innerHTML = sourceBadge(r.recommendation_source);
  document.getElementById('results-recommendations').innerHTML = buildRecommendationsHtml(
    recs,
    'No recommendations generated. Run an analysis first.'
  );

  // Narrative
  const nar = r.narrative_explanation || 'No narrative generated.';
  document.getElementById('results-narrative').innerHTML = renderNarrativeMarkdown(nar);

  // RAG panel
  document.getElementById('results-rag').innerHTML = buildRagHtml(r);

  // Execution logs
  const eLogs = r.execution_logs || [];
  const errs = r.errors || [];
  document.getElementById('results-exec-logs').innerHTML =
    eLogs.map(l => `<div>${l}</div>`).join('') +
    (errs.length ? '<div style="color:var(--rose);margin-top:.3rem">── Errors ──</div>' : '') +
    errs.map(e => `<div style="color:#fda4af">⚠ ${e}</div>`).join('');
}

// ═══════════════════════════════════════════════════════════════
// Dependencies page
// ═══════════════════════════════════════════════════════════════

function renderDependenciesPage(r) {
  document.getElementById('dep-graph-content').innerHTML = buildDependenciesHtml(r);
}

// ═══════════════════════════════════════════════════════════════
// Recommendations page
// ═══════════════════════════════════════════════════════════════

function renderRecommendationsPage(r) {
  document.getElementById('recs-source-badge').innerHTML = sourceBadge(r.recommendation_source);
  const recs = r.recommendations || [];
  document.getElementById('recs-list').innerHTML = buildRecommendationsHtml(
    recs,
    'No recommendations generated. Run an analysis first.'
  );
}

function getTimelineCurrentRunMeta() {
  if (!AppState.timeline.runs.length) return null;
  return AppState.timeline.runs[AppState.timeline.currentIndex] || null;
}

function getTimelinePreviousRunMeta() {
  if (!AppState.timeline.runs.length || AppState.timeline.currentIndex <= 0) return null;
  return AppState.timeline.runs[AppState.timeline.currentIndex - 1] || null;
}

function getTimelineCachedResult(runMeta) {
  if (!runMeta || !runMeta.run_id) return null;
  return AppState.timeline.runDetailsCache[runMeta.run_id]?.result || null;
}

async function ensureTimelineRunDetail(runMeta) {
  if (!runMeta || !runMeta.run_id || !AppState.timeline.org) return null;
  const cached = AppState.timeline.runDetailsCache[runMeta.run_id];
  if (cached) return cached;

  const url = `/api/results/org/${encodeURIComponent(AppState.timeline.org)}?run_id=${encodeURIComponent(runMeta.run_id)}`;
  const resp = await fetch(url);
  if (!resp.ok) return null;

  const payload = await resp.json();
  if (!payload || payload.error) return null;

  AppState.timeline.runDetailsCache[runMeta.run_id] = payload;
  return payload;
}

async function hydrateTimelineDetailsForCurrent() {
  const current = getTimelineCurrentRunMeta();
  const previous = getTimelinePreviousRunMeta();
  if (current) {
    await ensureTimelineRunDetail(current);
  }
  if (previous) {
    await ensureTimelineRunDetail(previous);
  }
}

async function hydrateTimelineDetailsWindow(windowSize = 8) {
  if (!AppState.timeline.runs.length) return;
  const size = Math.max(2, windowSize);
  const runs = AppState.timeline.runs.slice(Math.max(0, AppState.timeline.runs.length - size));
  await Promise.all(runs.map(runMeta => ensureTimelineRunDetail(runMeta).catch(() => null)));
}

async function setTimelineRunIndex(nextIndex) {
  if (!AppState.timeline.runs.length) return;
  const bounded = Math.max(0, Math.min(nextIndex, AppState.timeline.runs.length - 1));
  AppState.timeline.currentIndex = bounded;
  AppState.selectedRunId = getTimelineCurrentRunMeta()?.run_id || '';
  try {
    await hydrateTimelineDetailsForCurrent();
  } catch (err) {
    console.warn('Could not hydrate timeline run details', err);
  }
  derivePredictiveAlerts();
  syncDeepLinkState();
  if (AppState.currentPage === 'dashboard') {
    renderDashboard();
  }
}

function toggleTimelinePlayback() {
  if (AppState.timeline.runs.length < 2) {
    showToast('Need at least 2 runs for timeline playback', 'info');
    return;
  }

  if (AppState.timeline.playing) {
    stopTimelinePlayback();
    renderDashboard();
    return;
  }

  AppState.timeline.playing = true;
  AppState.timeline.timer = setInterval(() => {
    const next = (AppState.timeline.currentIndex + 1) % AppState.timeline.runs.length;
    setTimelineRunIndex(next).catch(() => {});
  }, 1900);
  renderDashboard();
}

function computeTimelineEvents(runs) {
  const events = [];
  for (let i = 1; i < runs.length; i++) {
    const prev = runs[i - 1]?.summary || {};
    const curr = runs[i]?.summary || {};

    if (typeof prev.health_score === 'number' && typeof curr.health_score === 'number') {
      const deltaHealth = curr.health_score - prev.health_score;
      if (deltaHealth <= -10) {
        events.push({
          id: `critical-drift-${i}`,
          index: i,
          severity: 'high',
          title: 'Critical health drift',
          message: `Health dropped ${Math.abs(deltaHealth).toFixed(1)} points vs previous run.`,
        });
      }
    }

    if (typeof prev.risk_count === 'number' && typeof curr.risk_count === 'number') {
      const deltaRisk = curr.risk_count - prev.risk_count;
      if (deltaRisk >= 2) {
        events.push({
          id: `risk-spike-${i}`,
          index: i,
          severity: deltaRisk >= 4 ? 'high' : 'medium',
          title: 'Risk spike',
          message: `Risk count increased by ${deltaRisk} since the previous run.`,
        });
      }
    }

    if (prev.health_status && curr.health_status && prev.health_status !== curr.health_status) {
      events.push({
        id: `status-shift-${i}`,
        index: i,
        severity: 'low',
        title: 'Health status changed',
        message: `${prev.health_status} → ${curr.health_status}.`,
      });
    }
  }
  return events;
}

function derivePredictiveAlerts() {
  const alerts = [];
  const currentMeta = getTimelineCurrentRunMeta();
  const previousMeta = getTimelinePreviousRunMeta();
  const currentResult = getTimelineCachedResult(currentMeta);
  const previousResult = getTimelineCachedResult(previousMeta);

  if (currentMeta && previousMeta) {
    const currentSummary = currentMeta.summary || {};
    const previousSummary = previousMeta.summary || {};

    if (typeof currentSummary.health_score === 'number' && typeof previousSummary.health_score === 'number') {
      const healthDrop = currentSummary.health_score - previousSummary.health_score;
      if (healthDrop <= -10) {
        alerts.push({
          id: 'alert-critical-drift',
          severity: 'high',
          title: 'Critical Drift Predicted',
          description: `Health has fallen ${Math.abs(healthDrop).toFixed(1)} points across consecutive runs.`,
          evidenceHint: 'Timeline delta and score-change factors show negative momentum.',
        });
      }
    }

    if (typeof currentSummary.risk_count === 'number' && typeof previousSummary.risk_count === 'number') {
      const riskSpike = currentSummary.risk_count - previousSummary.risk_count;
      if (riskSpike >= 2) {
        alerts.push({
          id: 'alert-stalled-spike',
          severity: riskSpike >= 4 ? 'high' : 'medium',
          title: 'Stalled Issues Spike',
          description: `Risk count rose by ${riskSpike}; probable blocker or review backlog accumulation.`,
          evidenceHint: 'Risk count growth and recent run risk composition.',
        });
      }
    }

    const currDep = asNumber(currentResult?.sprint_analysis?.dependency_risk_score, NaN);
    const prevDep = asNumber(previousResult?.sprint_analysis?.dependency_risk_score, NaN);
    if (Number.isFinite(currDep) && Number.isFinite(prevDep)) {
      const depRise = currDep - prevDep;
      if (depRise >= 10) {
        alerts.push({
          id: 'alert-dependency-risk',
          severity: depRise >= 18 ? 'high' : 'medium',
          title: 'Rising Dependency Risk',
          description: `Dependency risk increased by ${depRise.toFixed(1)} points over the previous run.`,
          evidenceHint: 'Dependency propagation trend in sprint analysis.',
        });
      }
    }
  }

  if (currentResult?.sprint_analysis && typeof currentResult.sprint_analysis.confidence_score === 'number') {
    const confidencePct = currentResult.sprint_analysis.confidence_score <= 1
      ? currentResult.sprint_analysis.confidence_score * 100
      : currentResult.sprint_analysis.confidence_score;
    if (confidencePct < 45) {
      alerts.push({
        id: 'alert-low-confidence',
        severity: 'medium',
        title: 'Low Confidence Forecast',
        description: `Current prediction confidence is ${confidencePct.toFixed(1)}%, so forecast volatility is elevated.`,
        evidenceHint: 'Confidence score and evidence density for the latest run.',
      });
    }
  }

  if (AppState.comparisonResults.length >= 2) {
    const healthValues = AppState.comparisonResults
      .map(item => item.health)
      .filter(v => Number.isFinite(v));
    if (healthValues.length >= 2) {
      const avgHealth = healthValues.reduce((a, b) => a + b, 0) / healthValues.length;
      const focusOrg = AppState.comparisonResults.find(item => item.organization === AppState.timeline.org)
        || AppState.comparisonResults[0];
      if (focusOrg && Number.isFinite(focusOrg.health) && focusOrg.health <= avgHealth - 12) {
        alerts.push({
          id: 'alert-cross-org-outlier',
          severity: 'medium',
          title: 'Cross-Org Performance Outlier',
          description: `${focusOrg.organization} trails multi-org mean by ${(avgHealth - focusOrg.health).toFixed(1)} points.`,
          evidenceHint: 'Comparison board variance against cohort mean.',
        });
      }
    }
  }

  AppState.alerts = alerts;
}

function toggleOrgForComparison(orgName, checked) {
  const selected = new Set(AppState.comparisonSelection);
  if (checked) {
    if (selected.size >= 6) {
      showToast('You can compare at most 6 organizations', 'info');
      renderDashboard();
      return;
    }
    selected.add(orgName);
  } else {
    selected.delete(orgName);
  }
  AppState.comparisonSelection = Array.from(selected);
  syncDeepLinkState();
}

async function loadMultiOrgComparison() {
  if (AppState.comparisonSelection.length < 2) {
    showToast('Select at least 2 organizations to compare', 'info');
    return;
  }

  const selected = AppState.comparisonSelection.slice(0, 6);
  const results = await Promise.all(selected.map(async orgName => {
    try {
      const resp = await fetch(`/api/results/org/${encodeURIComponent(orgName)}`);
      if (!resp.ok) return null;
      const payload = await resp.json();
      if (payload.error || !payload.result) return null;

      const analysis = payload.result.sprint_analysis || {};
      const rawConfidence = analysis.confidence_score;
      const confidence = typeof rawConfidence === 'number'
        ? (rawConfidence <= 1 ? rawConfidence * 100 : rawConfidence)
        : 0;

      return {
        organization: payload.organization || orgName,
        runId: payload.entry?.run_id || '',
        createdAt: payload.entry?.created_at || '',
        health: asNumber(analysis.health_score, NaN),
        completion: asNumber(analysis.completion_probability, NaN),
        riskCount: Array.isArray(payload.result.identified_risks)
          ? payload.result.identified_risks.length
          : asNumber(payload.entry?.summary?.risk_count, 0),
        confidence,
      };
    } catch (err) {
      console.warn('Failed comparison fetch for org', orgName, err);
      return null;
    }
  }));

  AppState.comparisonResults = results.filter(Boolean);
  derivePredictiveAlerts();
  syncDeepLinkState();
  renderDashboard();

  if (AppState.comparisonResults.length) {
    showToast(`Loaded comparison for ${AppState.comparisonResults.length} organizations`, 'success');
  } else {
    showToast('No comparison data available for selected organizations', 'error');
  }
}

function showAlertEvidence(alertId) {
  openExplainabilityDrawer('alert', alertId);
}

function closeExplainabilityDrawer() {
  const drawer = document.getElementById('explain-drawer');
  const backdrop = document.getElementById('explain-backdrop');
  if (drawer) {
    drawer.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
  }
  if (backdrop) {
    backdrop.classList.remove('open');
  }
}

async function openExplainabilityDrawer(mode = 'timeline', payload = '') {
  const drawer = document.getElementById('explain-drawer');
  const backdrop = document.getElementById('explain-backdrop');
  const titleEl = document.getElementById('explain-drawer-title');
  const contentEl = document.getElementById('explain-drawer-content');
  if (!drawer || !backdrop || !titleEl || !contentEl) return;

  try {
    await hydrateTimelineDetailsForCurrent();
  } catch (err) {
    console.warn('Could not load explainability context', err);
  }

  const currentMeta = getTimelineCurrentRunMeta();
  const previousMeta = getTimelinePreviousRunMeta();
  const currentResult = getTimelineCachedResult(currentMeta) || AppState.lastResult;
  const previousResult = getTimelineCachedResult(previousMeta);

  let title = 'Explainability';
  let body = '<div>No explainability data available.</div>';

  if (mode === 'alert') {
    const alert = AppState.alerts.find(item => item.id === payload);
    title = alert ? `Evidence: ${alert.title}` : 'Alert Evidence';
    const citations = Array.from(new Set([
      ...(currentResult?.evidence_citations || []),
      ...(currentResult?.rag_context?.evidence_citations || []),
    ].filter(Boolean)));

    body = `
      <div><strong>${escapeHtml(alert?.title || 'Alert')}</strong></div>
      <div style="margin-top:.25rem">${escapeHtml(alert?.description || 'No alert details found.')}</div>
      <div style="margin-top:.4rem;color:var(--text-muted)">${escapeHtml(alert?.evidenceHint || '')}</div>
      <div style="margin-top:.75rem"><strong>Source Links / Citations</strong></div>
      <div class="citation-list">
        ${citations.length ? citations.map(item => {
          const text = escapeHtml(item);
          if (/^https?:\/\//i.test(item)) {
            return `<div class="citation-item"><a href="${text}" target="_blank" rel="noopener noreferrer">${text}</a></div>`;
          }
          return `<div class="citation-item">${text}</div>`;
        }).join('') : '<div class="citation-item">No citation links found for this alert.</div>'}
      </div>
    `;
  } else {
    title = 'Why Score Changed';

    if (!currentResult || !previousResult) {
      body = '<div>At least two timeline runs are required for ranked factor explainability.</div>';
    } else {
      const currentAnalysis = currentResult.sprint_analysis || {};
      const prevAnalysis = previousResult.sprint_analysis || {};

      const metrics = [
        { key: 'delivery_score', label: 'Delivery', invert: false },
        { key: 'momentum_score', label: 'Momentum', invert: false },
        { key: 'quality_score', label: 'Quality', invert: false },
        { key: 'collaboration_score', label: 'Collaboration', invert: false },
        { key: 'completion_probability', label: 'Completion Probability', invert: false },
        { key: 'dependency_risk_score', label: 'Dependency Risk', invert: true },
      ];

      const ranked = metrics.map(metric => {
        const curr = asNumber(currentAnalysis[metric.key], NaN);
        const prev = asNumber(prevAnalysis[metric.key], NaN);
        if (!Number.isFinite(curr) || !Number.isFinite(prev)) return null;
        const rawDelta = curr - prev;
        const impactDelta = metric.invert ? -rawDelta : rawDelta;
        return {
          ...metric,
          rawDelta,
          impactDelta,
          magnitude: Math.abs(rawDelta),
        };
      }).filter(Boolean)
        .sort((a, b) => b.magnitude - a.magnitude);

      const citations = Array.from(new Set([
        ...(currentResult.evidence_citations || []),
        ...(currentResult.rag_context?.evidence_citations || []),
      ].filter(Boolean)));

      body = `
        <div>
          Comparing ${escapeHtml(previousMeta?.run_id || 'previous run')} → ${escapeHtml(currentMeta?.run_id || 'current run')}
        </div>
        <div class="factor-list">
          ${ranked.length ? ranked.map(item => {
            const direction = item.impactDelta >= 0 ? '↑' : '↓';
            const deltaText = `${direction} ${Math.abs(item.rawDelta).toFixed(1)}`;
            const width = Math.max(6, Math.min(100, item.magnitude));
            return `<div class="factor-item">
              <div class="factor-head"><span>${escapeHtml(item.label)}</span><span>${deltaText}</span></div>
              <div class="factor-bar"><span style="width:${width}%"></span></div>
            </div>`;
          }).join('') : '<div class="factor-item">No comparable metrics found across the selected runs.</div>'}
        </div>
        <div><strong>Source Links / Citations</strong></div>
        <div class="citation-list">
          ${citations.length ? citations.map(item => {
            const text = escapeHtml(item);
            if (/^https?:\/\//i.test(item)) {
              return `<div class="citation-item"><a href="${text}" target="_blank" rel="noopener noreferrer">${text}</a></div>`;
            }
            return `<div class="citation-item">${text}</div>`;
          }).join('') : '<div class="citation-item">No citation links found for these runs.</div>'}
        </div>
      `;
    }
  }

  titleEl.textContent = title;
  contentEl.innerHTML = body;

  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  backdrop.classList.add('open');
}

function buildOrganizationRetrieveCard() {
  const organizations = AppState.organizationIndex || [];
  const selectedOrg = AppState.selectedOrganization;
  const selectedMeta = organizations.find(item => item.organization === selectedOrg) || null;
  const filteredRuns = getFilteredOrgRuns();

  const options = organizations.length
    ? organizations.map(item => {
      const isSelected = item.organization === selectedOrg ? 'selected' : '';
      return `<option value="${escapeHtml(item.organization)}" ${isSelected}>${escapeHtml(item.organization)} (${item.run_count || 0})</option>`;
    }).join('')
    : '<option value="">No organizations with recorded inference</option>';

  const latestStamp = selectedMeta?.latest_timestamp
    ? `${formatRelativeTime(selectedMeta.latest_timestamp)} (${new Date(selectedMeta.latest_timestamp).toLocaleString()})`
    : 'No runs recorded yet';

  const runItems = filteredRuns.length
    ? filteredRuns.map(run => {
      const summary = run.summary || {};
      const health = typeof summary.health_score === 'number' ? summary.health_score.toFixed(1) : 'n/a';
      const completion = typeof summary.completion_probability === 'number' ? `${summary.completion_probability.toFixed(0)}%` : 'n/a';
      const isActive = AppState.selectedRunId && AppState.selectedRunId === run.run_id ? 'active' : '';
      const encodedRunId = encodeURIComponent(run.run_id || '');
      return `<div class="org-run-item ${isActive}">
        <div class="org-run-meta">
          <div class="org-run-main">Run ${escapeHtml(run.run_id || 'unknown')}</div>
          <div class="org-run-sub">${escapeHtml(formatRelativeTime(run.created_at))} · ${escapeHtml(run.source || 'analyze')} · ${escapeHtml(run.eval_mode || 'resilient')}</div>
          <div class="org-run-kpis">
            <span class="org-pill">Health ${health}</span>
            <span class="org-pill">Completion ${completion}</span>
            <span class="org-pill">Risks ${summary.risk_count ?? 0}</span>
          </div>
        </div>
        <div class="org-run-actions">
          <button class="btn btn-ghost btn-sm" onclick="retrieveOrganizationResult(decodeURIComponent('${encodedRunId}'))">Load Run</button>
        </div>
      </div>`;
    }).join('')
    : '<div class="org-empty">Select an organization and retrieve recorded inference runs.</div>';

  return `
    <div class="card org-retrieve-card" style="margin-bottom:1.25rem">
      <div class="card-title"><span class="icon">🧭</span> Retrieve Recorded Inference By Organization</div>
      <div class="org-retrieve-grid">
        <div class="form-group" style="margin:0">
          <label for="org-select">Organization</label>
          <select id="org-select" onchange="onOrganizationChange(this.value)">
            ${options}
          </select>
        </div>
        <div class="btn-row">
          <button class="btn btn-secondary" onclick="refreshOrganizationIndex().then(() => renderDashboard())">↻ Refresh</button>
          <button class="btn btn-primary" onclick="retrieveOrganizationResult()" ${organizations.length ? '' : 'disabled'}>Retrieve Latest</button>
        </div>
      </div>
      <div class="org-meta-row">
        <span class="org-pill">${selectedMeta ? `${selectedMeta.run_count || 0} runs` : '0 runs'}</span>
        <span class="org-pill">${filteredRuns.length} filtered</span>
        <span class="org-meta-text">${escapeHtml(latestStamp)}</span>
      </div>
      <div class="org-run-list">
        ${runItems}
      </div>
    </div>
  `;
}

function buildSavedViewsPanel() {
  const organizations = AppState.organizationIndex || [];
  const orgOptions = organizations.length
    ? organizations.map(item => {
      const selected = (AppState.filters.organization || '') === item.organization ? 'selected' : '';
      return `<option value="${escapeHtml(item.organization)}" ${selected}>${escapeHtml(item.organization)}</option>`;
    }).join('')
    : '<option value="">No organizations</option>';

  const sourceOptions = [
    { value: 'all', label: 'Any Source' },
    { value: 'analyze', label: 'Analyze' },
    { value: 'analyze_sprint', label: 'Analyze Sprint JSON' },
    { value: 'analyze_query', label: 'Analyze Query' },
    { value: 'analyze_mock', label: 'Analyze Mock' },
  ];

  const presets = Object.entries(AppState.filterPresets || {})
    .sort(([, a], [, b]) => String(b?.updatedAt || '').localeCompare(String(a?.updatedAt || '')));

  return `
    <div class="card" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">🗂️</span> Saved Views & Filters</div>
      <div class="saved-views-row">
        <div class="form-group" style="margin:0">
          <label for="view-filter-org">Organization</label>
          <select id="view-filter-org">${orgOptions}</select>
        </div>
        <div class="form-group" style="margin:0">
          <label for="view-filter-date">Date Range</label>
          <select id="view-filter-date">
            <option value="all" ${AppState.filters.dateRange === 'all' ? 'selected' : ''}>All Time</option>
            <option value="7" ${AppState.filters.dateRange === '7' ? 'selected' : ''}>Last 7 days</option>
            <option value="14" ${AppState.filters.dateRange === '14' ? 'selected' : ''}>Last 14 days</option>
            <option value="30" ${AppState.filters.dateRange === '30' ? 'selected' : ''}>Last 30 days</option>
            <option value="90" ${AppState.filters.dateRange === '90' ? 'selected' : ''}>Last 90 days</option>
          </select>
        </div>
        <div class="form-group" style="margin:0">
          <label for="view-filter-risk">Risk Threshold</label>
          <select id="view-filter-risk">
            <option value="all" ${AppState.filters.riskThreshold === 'all' ? 'selected' : ''}>Any</option>
            <option value="1" ${AppState.filters.riskThreshold === '1' ? 'selected' : ''}>1+</option>
            <option value="2" ${AppState.filters.riskThreshold === '2' ? 'selected' : ''}>2+</option>
            <option value="3" ${AppState.filters.riskThreshold === '3' ? 'selected' : ''}>3+</option>
            <option value="5" ${AppState.filters.riskThreshold === '5' ? 'selected' : ''}>5+</option>
          </select>
        </div>
        <div class="form-group" style="margin:0">
          <label for="view-filter-source">Source Mode</label>
          <select id="view-filter-source">
            ${sourceOptions.map(source => `<option value="${source.value}" ${AppState.filters.sourceMode === source.value ? 'selected' : ''}>${source.label}</option>`).join('')}
          </select>
        </div>
      </div>

      <div class="btn-row" style="margin-bottom:.4rem">
        <button class="btn btn-secondary" onclick="applyViewFiltersFromControls()">Apply Filters</button>
        <button class="btn btn-ghost" onclick="resetViewFilters()">Reset</button>
      </div>

      <div class="saved-views-preset-row">
        <input id="preset-name-input" type="text" placeholder="Preset name (e.g. weekly-risk-watch)" />
        <button class="btn btn-primary" onclick="saveCurrentFilterPreset()">Save Preset</button>
      </div>

      <div class="saved-views-list">
        ${presets.length ? presets.map(([name]) => {
          const encodedName = encodeURIComponent(name);
          return `<div class="preset-chip">
            <span>${escapeHtml(name)}</span>
            <button title="Apply preset" onclick="applyFilterPreset(decodeURIComponent('${encodedName}'))">↺</button>
            <button title="Delete preset" onclick="deleteFilterPreset(decodeURIComponent('${encodedName}'))">✕</button>
          </div>`;
        }).join('') : '<span class="org-meta-text">No presets saved yet.</span>'}
      </div>
    </div>
  `;
}

function buildMultiOrgComparisonBoard() {
  const organizations = AppState.organizationIndex || [];

  const selector = organizations.length
    ? organizations.map(item => {
      const encodedOrg = encodeURIComponent(item.organization);
      const checked = AppState.comparisonSelection.includes(item.organization) ? 'checked' : '';
      return `<label class="compare-check">
        <input type="checkbox" ${checked} onchange="toggleOrgForComparison(decodeURIComponent('${encodedOrg}'), this.checked)">
        <span>${escapeHtml(item.organization)}</span>
      </label>`;
    }).join('')
    : '<div class="org-empty">No organizations available for comparison.</div>';

  const healthValues = AppState.comparisonResults.map(item => item.health).filter(v => Number.isFinite(v));
  const completionValues = AppState.comparisonResults.map(item => item.completion).filter(v => Number.isFinite(v));
  const avgHealth = healthValues.length ? healthValues.reduce((a, b) => a + b, 0) / healthValues.length : NaN;
  const avgCompletion = completionValues.length ? completionValues.reduce((a, b) => a + b, 0) / completionValues.length : NaN;

  const cards = AppState.comparisonResults.length
    ? AppState.comparisonResults.map(item => {
      const healthDiff = Number.isFinite(avgHealth) && Number.isFinite(item.health) ? item.health - avgHealth : 0;
      const completionDiff = Number.isFinite(avgCompletion) && Number.isFinite(item.completion)
        ? item.completion - avgCompletion
        : 0;
      const varianceClass = healthDiff > 1 ? 'positive' : healthDiff < -1 ? 'negative' : 'neutral';
      const varianceSign = healthDiff > 0 ? '↑' : healthDiff < 0 ? '↓' : '→';
      const confidence = Math.max(0, Math.min(100, asNumber(item.confidence, 0)));
      const completionText = Number.isFinite(item.completion) ? `${item.completion.toFixed(0)}%` : 'n/a';
      const healthText = Number.isFinite(item.health) ? item.health.toFixed(1) : 'n/a';

      return `<div class="compare-card">
        <div class="org-name">${escapeHtml(item.organization)}</div>
        <div class="org-meta">${escapeHtml(formatRelativeTime(item.createdAt))}</div>
        <div class="compare-kpis">
          <div class="compare-kpi"><div class="k">Health</div><div class="v">${healthText}</div></div>
          <div class="compare-kpi"><div class="k">Completion</div><div class="v">${completionText}</div></div>
          <div class="compare-kpi"><div class="k">Risks</div><div class="v">${item.riskCount}</div></div>
          <div class="compare-kpi"><div class="k">Comp Δ</div><div class="v">${completionDiff >= 0 ? '+' : ''}${completionDiff.toFixed(1)}</div></div>
        </div>
        <div class="variance-chip ${varianceClass}">${varianceSign} ${Math.abs(healthDiff).toFixed(1)} vs cohort mean</div>
        <div class="confidence-wrap">
          <div class="confidence-label"><span>Confidence Ribbon</span><span>${confidence.toFixed(0)}%</span></div>
          <div class="confidence-ribbon"><span style="width:${confidence}%"></span></div>
        </div>
      </div>`;
    }).join('')
    : '<div class="org-empty">Select 2-6 organizations and click Compare Selected.</div>';

  return `
    <div class="card" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">📚</span> Multi-Org Comparison Board</div>
      <div class="comparison-toolbar">
        <div class="compare-select-grid">${selector}</div>
        <div class="btn-row" style="align-items:center">
          <button class="btn btn-primary" onclick="loadMultiOrgComparison()">Compare Selected</button>
          <span style="font-size:.64rem;color:var(--text-muted)">Pick 2 to 6 organizations</span>
        </div>
      </div>
      <div class="compare-grid">${cards}</div>
    </div>
  `;
}

function buildTimelinePlaybackPanel() {
  const organizations = AppState.organizationIndex || [];
  const selectedOrg = AppState.timeline.org || AppState.selectedOrganization || '';
  const options = organizations.length
    ? organizations.map(item => {
      const selected = item.organization === selectedOrg ? 'selected' : '';
      return `<option value="${escapeHtml(item.organization)}" ${selected}>${escapeHtml(item.organization)}</option>`;
    }).join('')
    : '<option value="">No organization</option>';

  if (!selectedOrg || !AppState.timeline.runs.length) {
    return `
      <div class="card" style="margin-bottom:1rem">
        <div class="card-title"><span class="icon">⏱️</span> Timeline Playback</div>
        <div class="form-row" style="margin-bottom:.55rem">
          <div class="form-group" style="margin:0">
            <label>Timeline Organization</label>
            <select onchange="onOrganizationChange(this.value)">${options}</select>
          </div>
          <div class="btn-row" style="align-items:end">
            <button class="btn btn-secondary" onclick="loadOrganizationHistory(AppState.selectedOrganization, { rerender: true })">Load Timeline</button>
          </div>
        </div>
        <div class="org-empty">No timeline runs loaded for this organization yet.</div>
      </div>
    `;
  }

  const currentMeta = getTimelineCurrentRunMeta();
  const currentSummary = currentMeta?.summary || {};
  const events = computeTimelineEvents(AppState.timeline.runs);
  const currentEvents = events.filter(item => item.index === AppState.timeline.currentIndex);
  const completionText = typeof currentSummary.completion_probability === 'number'
    ? `${currentSummary.completion_probability.toFixed(0)}%`
    : 'n/a';

  return `
    <div class="card" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">⏱️</span> Timeline Playback for Sprint Runs</div>
      <div class="timeline-controls">
        <div>
          <div class="form-group" style="margin:0 0 .4rem 0">
            <label>Timeline Organization</label>
            <select onchange="onOrganizationChange(this.value)">${options}</select>
          </div>
          <input
            class="timeline-range"
            type="range"
            min="0"
            max="${AppState.timeline.runs.length - 1}"
            value="${AppState.timeline.currentIndex}"
            oninput="setTimelineRunIndex(Number(this.value))"
          >
        </div>
        <div class="btn-row">
          <button class="btn btn-secondary" onclick="toggleTimelinePlayback()">${AppState.timeline.playing ? 'Pause' : 'Play'} Timeline</button>
          <button class="btn btn-ghost" onclick="openExplainabilityDrawer('timeline')">Why Score Changed</button>
        </div>
      </div>

      <div class="timeline-board">
        <div class="timeline-stage">
          <div class="run-title">Run ${escapeHtml(currentMeta?.run_id || 'unknown')}</div>
          <div class="run-sub">${escapeHtml(formatRelativeTime(currentMeta?.created_at))} · Source: ${escapeHtml(currentMeta?.source || 'analyze')}</div>
          <div class="org-run-kpis" style="margin-top:.32rem">
            <span class="org-pill">Health ${typeof currentSummary.health_score === 'number' ? currentSummary.health_score.toFixed(1) : 'n/a'}</span>
            <span class="org-pill">Completion ${completionText}</span>
            <span class="org-pill">Risks ${currentSummary.risk_count ?? 0}</span>
          </div>
        </div>
        <div class="timeline-events">
          ${currentEvents.length ? currentEvents.map(eventItem => `<div class="timeline-event-item ${eventItem.severity}"><span>${escapeHtml(eventItem.title)}</span><span>${escapeHtml(eventItem.message)}</span></div>`).join('') : '<div class="org-empty" style="margin-top:.4rem">No event markers on this run.</div>'}
        </div>
      </div>
    </div>
  `;
}

function buildPredictiveAlertsPanel() {
  const alertRows = AppState.alerts.length
    ? AppState.alerts.map(item => {
      const encoded = encodeURIComponent(item.id);
      return `<div class="alert-item ${item.severity}">
        <div class="alert-head">
          <div class="alert-title">${escapeHtml(item.title)}</div>
          <span class="badge badge-${item.severity === 'high' ? 'high' : item.severity === 'medium' ? 'medium' : 'info'}">${escapeHtml(item.severity)}</span>
        </div>
        <div class="alert-desc">${escapeHtml(item.description)}</div>
        <div class="alert-meta">${escapeHtml(item.evidenceHint || '')}</div>
        <div class="btn-row" style="margin-top:.35rem">
          <button class="btn btn-ghost btn-sm" onclick="showAlertEvidence(decodeURIComponent('${encoded}'))">Show Evidence</button>
        </div>
      </div>`;
    }).join('')
    : '<div class="org-empty">No predictive alerts detected yet. Load timeline or comparison data to evaluate alert rules.</div>';

  return `
    <div class="card" style="margin-bottom:1.1rem">
      <div class="card-title"><span class="icon">🚨</span> Predictive Alerts Panel</div>
      <div class="alerts-list">${alertRows}</div>
    </div>
  `;
}

function flattenFeatureMetrics(resultPayload) {
  const rows = [];
  const features = resultPayload?.features || {};

  Object.entries(features).forEach(([category, metrics]) => {
    if (!metrics || typeof metrics !== 'object') return;
    Object.entries(metrics).forEach(([metric, rawValue]) => {
      const value = Number(rawValue);
      if (!Number.isFinite(value)) return;
      rows.push({
        key: `${category}.${metric}`,
        category,
        metric,
        metricLabel: metric.replace(/_/g, ' '),
        value,
      });
    });
  });

  return rows;
}

function getPreviousResultForCurrentRun() {
  const previousMeta = getTimelinePreviousRunMeta();
  const previousResult = getTimelineCachedResult(previousMeta);
  if (previousResult) return previousResult;

  const timelineRuns = AppState.timeline.runs;
  if (!timelineRuns.length) return null;

  const activeRunId = AppState.selectedRunId || getTimelineCurrentRunMeta()?.run_id || '';
  if (!activeRunId) return null;

  const currentIndex = timelineRuns.findIndex(item => item.run_id === activeRunId);
  if (currentIndex > 0) {
    return getTimelineCachedResult(timelineRuns[currentIndex - 1]);
  }

  return null;
}

function buildMetricDrilldownRows(resultPayload) {
  const currentRows = flattenFeatureMetrics(resultPayload);
  if (!currentRows.length) return [];

  const previousResult = getPreviousResultForCurrentRun();
  const previousRows = flattenFeatureMetrics(previousResult);
  const previousMap = new Map(previousRows.map(item => [item.key, item.value]));

  const distributionMap = {};
  AppState.timeline.runs.forEach(meta => {
    const result = getTimelineCachedResult(meta);
    if (!result) return;
    flattenFeatureMetrics(result).forEach(row => {
      if (!Array.isArray(distributionMap[row.key])) {
        distributionMap[row.key] = [];
      }
      distributionMap[row.key].push(row.value);
    });
  });

  currentRows.forEach(row => {
    if (!Array.isArray(distributionMap[row.key])) {
      distributionMap[row.key] = [row.value];
    }
  });

  return currentRows.map(row => {
    const previousValue = previousMap.has(row.key) ? previousMap.get(row.key) : NaN;
    const delta = Number.isFinite(previousValue) ? row.value - previousValue : NaN;
    const sigma = stddev(distributionMap[row.key] || []);
    const zDelta = Number.isFinite(delta) && Number.isFinite(sigma) && sigma > 0
      ? delta / sigma
      : NaN;

    return {
      ...row,
      previousValue,
      delta,
      zDelta,
      sampleSize: (distributionMap[row.key] || []).length,
    };
  });
}

function sortDrilldownRows(rows) {
  const sortKey = AppState.drilldown.sortKey || DRILLDOWN_DEFAULT_SORT.key;
  const sortDir = AppState.drilldown.sortDir || DRILLDOWN_DEFAULT_SORT.direction;
  const direction = sortDir === 'asc' ? 1 : -1;

  const normalizeNumeric = value => {
    if (Number.isFinite(value)) return value;
    return sortDir === 'asc' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
  };

  return [...rows].sort((a, b) => {
    if (sortKey === 'category') {
      return direction * String(a.category).localeCompare(String(b.category));
    }

    if (sortKey === 'metric') {
      return direction * String(a.metricLabel).localeCompare(String(b.metricLabel));
    }

    if (sortKey === 'value') {
      return direction * (normalizeNumeric(a.value) - normalizeNumeric(b.value));
    }

    if (sortKey === 'delta') {
      return direction * (normalizeNumeric(a.delta) - normalizeNumeric(b.delta));
    }

    return direction * (normalizeNumeric(a.zDelta) - normalizeNumeric(b.zDelta));
  });
}

function sortIndicator(key) {
  if (AppState.drilldown.sortKey !== key) return '↕';
  return AppState.drilldown.sortDir === 'asc' ? '↑' : '↓';
}

function formatSignedNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value).toFixed(digits);
  return `${value >= 0 ? '+' : '-'}${abs}`;
}

function buildSparkBars(points, key, maxValue) {
  if (!points.length) {
    return '<div class="sparkline-empty">No trend data</div>';
  }

  const values = points
    .map(item => Number(item[key]))
    .filter(v => Number.isFinite(v));

  if (!values.length) {
    return '<div class="sparkline-empty">No trend data</div>';
  }

  const baseline = maxValue || Math.max(...values, 1);

  return `<div class="sparkline">${points.map(point => {
    const raw = Number(point[key]);
    const ratio = Number.isFinite(raw) ? Math.max(0.06, Math.min(1, raw / baseline)) : 0.06;
    const label = Number.isFinite(raw) ? raw.toFixed(2) : 'n/a';
    return `<span style="height:${(ratio * 100).toFixed(1)}%" title="${escapeHtml(point.runId || '')}: ${label}"></span>`;
  }).join('')}</div>`;
}

function computeEvidenceScorecard() {
  const points = [];

  AppState.timeline.runs.forEach(meta => {
    const payload = getTimelineCachedResult(meta);
    if (!payload) return;

    const runMetrics = payload.run_metrics || {};
    const citationQuality = runMetrics.citation_quality || {};
    const citationScore = asNumber(citationQuality.score, NaN);
    const totalCitations = asNumber(citationQuality.total_citations, NaN);
    const nonEmptyCitations = asNumber(citationQuality.non_empty_citations, NaN);
    const coverage = Number.isFinite(totalCitations) && totalCitations > 0
      ? (nonEmptyCitations / totalCitations) * 100
      : NaN;

    points.push({
      runId: meta.run_id,
      createdAt: meta.created_at,
      citationScore,
      coverage,
      reliability: computeSourceReliabilityScore(runMetrics.source_breakdown || {}),
    });
  });

  if (!points.length && AppState.lastResult) {
    const runMetrics = AppState.lastResult.run_metrics || {};
    const citationQuality = runMetrics.citation_quality || {};
    const totalCitations = asNumber(citationQuality.total_citations, NaN);
    const nonEmptyCitations = asNumber(citationQuality.non_empty_citations, NaN);
    const coverage = Number.isFinite(totalCitations) && totalCitations > 0
      ? (nonEmptyCitations / totalCitations) * 100
      : NaN;

    points.push({
      runId: AppState.selectedRunId || 'current',
      createdAt: new Date().toISOString(),
      citationScore: asNumber(citationQuality.score, NaN),
      coverage,
      reliability: computeSourceReliabilityScore(runMetrics.source_breakdown || {}),
    });
  }

  if (!points.length) {
    return {
      points: [],
      current: null,
      previous: null,
      citationDelta: NaN,
      reliabilityDelta: NaN,
      coverageDelta: NaN,
    };
  }

  const activeRunId = AppState.selectedRunId || getTimelineCurrentRunMeta()?.run_id || '';
  let currentIndex = points.length - 1;
  if (activeRunId) {
    const matched = points.findIndex(point => point.runId === activeRunId);
    if (matched >= 0) currentIndex = matched;
  }

  const current = points[currentIndex] || points[points.length - 1];
  const previous = currentIndex > 0 ? points[currentIndex - 1] : null;

  return {
    points,
    current,
    previous,
    citationDelta: current && previous ? current.citationScore - previous.citationScore : NaN,
    reliabilityDelta: current && previous ? current.reliability - previous.reliability : NaN,
    coverageDelta: current && previous ? current.coverage - previous.coverage : NaN,
  };
}

function buildExportCenterPanel() {
  const metadata = buildReproducibilityMetadata();
  const shareUrl = buildShareableDashboardUrl();

  return `
    <div class="card accent-inference" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">📤</span> Export Center</div>
      <div class="export-meta-grid">
        <span class="org-pill">Run ${escapeHtml(metadata.run_id || 'latest')}</span>
        <span class="org-pill">Eval ${escapeHtml(metadata.eval_mode || 'resilient')}</span>
        <span class="org-pill">Input ${escapeHtml(metadata.input_mode || 'json')}</span>
        <span class="org-pill">Fingerprint ${escapeHtml(metadata.reproducibility_fingerprint || '')}</span>
      </div>
      <div class="btn-row" style="margin:.7rem 0 .55rem">
        <button class="btn btn-secondary" onclick="exportCurrentRun('json')">Export JSON</button>
        <button class="btn btn-secondary" onclick="exportCurrentRun('md')">Export Markdown</button>
        <button class="btn btn-primary" onclick="exportCurrentRun('pdf')">Export PDF</button>
        <button class="btn btn-ghost" onclick="copyDashboardDeepLink()">Copy Shared Link</button>
      </div>
      <div class="share-link-preview" title="${escapeHtml(shareUrl)}">${escapeHtml(shareUrl)}</div>
    </div>
  `;
}

function buildExperimentNotesPanel() {
  const record = getExperimentRecordForCurrentRun();
  const tags = Array.isArray(record.tags) ? record.tags.join(', ') : '';
  const lastUpdated = record.updatedAt ? `Last updated ${formatRelativeTime(record.updatedAt)}` : 'No experiment notes saved yet';

  return `
    <div class="card accent-inference" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">🧪</span> Experiment Tags & Notes</div>
      <div class="form-group" style="margin-bottom:.55rem">
        <label for="experiment-tags-input">Tags</label>
        <input id="experiment-tags-input" type="text" placeholder="baseline, strict-mode, lora-v2" value="${escapeHtml(tags)}" />
      </div>
      <div class="form-group" style="margin-bottom:.55rem">
        <label for="experiment-note-input">Analyst Notes</label>
        <textarea id="experiment-note-input" rows="4" placeholder="Record hypothesis, anomalies, or reproducibility notes...">${escapeHtml(record.note || '')}</textarea>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" onclick="saveExperimentNotes()">Save Notes</button>
      </div>
      <div class="org-meta-text" style="margin-top:.45rem">${escapeHtml(lastUpdated)}</div>
    </div>
  `;
}

function buildEvidenceQualityScorecard() {
  const scorecard = computeEvidenceScorecard();
  AppState.evidenceScorecard = scorecard;

  const current = scorecard.current;
  if (!current) {
    return `
      <div class="card accent-evidence" style="margin-bottom:1rem">
        <div class="card-title"><span class="icon">🧲</span> Evidence Quality Scorecard</div>
        <div class="org-empty">No evidence trend data available. Load timeline runs to compute citation and reliability trends.</div>
      </div>
    `;
  }

  const citation = Number.isFinite(current.citationScore) ? current.citationScore.toFixed(2) : 'n/a';
  const reliability = Number.isFinite(current.reliability) ? `${current.reliability.toFixed(1)}%` : 'n/a';
  const coverage = Number.isFinite(current.coverage) ? `${current.coverage.toFixed(1)}%` : 'n/a';

  return `
    <div class="card accent-evidence" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">🧲</span> Evidence Quality Scorecard</div>
      <div class="scorecard-grid">
        <div class="score-pill">
          <div class="k">Citation Quality</div>
          <div class="v">${citation}</div>
          <div class="delta ${scorecard.citationDelta >= 0 ? 'pos' : 'neg'}">${formatSignedNumber(scorecard.citationDelta, 2)} vs previous</div>
        </div>
        <div class="score-pill">
          <div class="k">Source Reliability</div>
          <div class="v">${reliability}</div>
          <div class="delta ${scorecard.reliabilityDelta >= 0 ? 'pos' : 'neg'}">${formatSignedNumber(scorecard.reliabilityDelta, 1)} vs previous</div>
        </div>
        <div class="score-pill">
          <div class="k">Citation Coverage</div>
          <div class="v">${coverage}</div>
          <div class="delta ${scorecard.coverageDelta >= 0 ? 'pos' : 'neg'}">${formatSignedNumber(scorecard.coverageDelta, 1)} vs previous</div>
        </div>
      </div>
      <div class="evidence-trends">
        <div class="trend-block">
          <div class="trend-head">Citation Trend</div>
          ${buildSparkBars(scorecard.points, 'citationScore', 1)}
        </div>
        <div class="trend-block">
          <div class="trend-head">Reliability Trend</div>
          ${buildSparkBars(scorecard.points, 'reliability', 100)}
        </div>
      </div>
    </div>
  `;
}

function buildMetricDrilldownPanel(resultPayload) {
  const allRows = buildMetricDrilldownRows(resultPayload);
  const query = (AppState.drilldown.query || '').trim().toLowerCase();
  const searchedRows = query
    ? allRows.filter(row => `${row.category} ${row.metricLabel}`.toLowerCase().includes(query))
    : allRows;
  const rows = sortDrilldownRows(searchedRows);

  const body = rows.length
    ? rows.slice(0, 160).map(row => {
      const deltaClass = Number.isFinite(row.delta)
        ? (row.delta >= 0 ? 'pos' : 'neg')
        : '';
      const zClass = Number.isFinite(row.zDelta)
        ? (row.zDelta >= 0 ? 'pos' : 'neg')
        : '';

      return `<tr>
        <td>${escapeHtml(row.category)}</td>
        <td>${escapeHtml(row.metricLabel)}</td>
        <td>${row.value.toFixed(3)}</td>
        <td>${Number.isFinite(row.previousValue) ? row.previousValue.toFixed(3) : '—'}</td>
        <td class="${deltaClass}">${formatSignedNumber(row.delta, 3)}</td>
        <td class="${zClass}">${formatSignedNumber(row.zDelta, 2)}</td>
      </tr>`;
    }).join('')
    : '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">No feature metrics match the current filter.</td></tr>';

  return `
    <div class="card accent-inference" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">🧮</span> Metric Drilldown (Feature Delta + Z-Score)</div>
      <div class="drilldown-toolbar">
        <input
          type="text"
          value="${escapeHtml(AppState.drilldown.query || '')}"
          placeholder="Search metric or category"
          oninput="setDrilldownSearch(this.value)"
        />
        <span class="org-meta-text">${rows.length} metrics shown</span>
      </div>
      <div class="drilldown-table-wrap">
        <table class="drilldown-table">
          <thead>
            <tr>
              <th><button onclick="setDrilldownSort('category')">Category ${sortIndicator('category')}</button></th>
              <th><button onclick="setDrilldownSort('metric')">Metric ${sortIndicator('metric')}</button></th>
              <th><button onclick="setDrilldownSort('value')">Current ${sortIndicator('value')}</button></th>
              <th>Previous</th>
              <th><button onclick="setDrilldownSort('delta')">Delta ${sortIndicator('delta')}</button></th>
              <th><button onclick="setDrilldownSort('z_delta')">Z-Delta ${sortIndicator('z_delta')}</button></th>
            </tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </div>
  `;
}

function buildTeamAnnotationLayer(risks = [], recs = []) {
  const records = getTeamAnnotationsForCurrentRun();
  const targets = [
    ...risks.slice(0, 8).map((item, index) => ({
      value: `risk:${index + 1}:${item.risk_type || 'risk'}`,
      label: `Risk · ${(item.risk_type || 'risk').replace(/_/g, ' ')}`,
    })),
    ...recs.slice(0, 8).map((item, index) => ({
      value: `rec:${index + 1}:${item.title || 'recommendation'}`,
      label: `Recommendation · ${item.title || 'recommendation'}`,
    })),
  ];

  const options = targets.length
    ? targets.map(target => `<option value="${escapeHtml(target.value)}">${escapeHtml(target.label)}</option>`).join('')
    : '';

  const list = records.length
    ? records.map(item => {
      const encodedId = encodeURIComponent(item.id || '');
      return `<div class="annotation-item">
        <div class="annotation-head">
          <span class="badge badge-${item.decision === 'escalate' ? 'high' : item.decision === 'accept' ? 'low' : 'info'}">${escapeHtml(item.decision || 'note')}</span>
          <span class="annotation-meta">${escapeHtml(item.author || 'analyst')} · ${escapeHtml(formatRelativeTime(item.createdAt))}</span>
        </div>
        <div class="annotation-target">${escapeHtml(item.target || 'general')}</div>
        <div class="annotation-comment">${escapeHtml(item.comment || '')}</div>
        <div class="btn-row" style="margin-top:.4rem">
          <button class="btn btn-ghost btn-sm" onclick="deleteTeamAnnotation(decodeURIComponent('${encodedId}'))">Remove</button>
        </div>
      </div>`;
    }).join('')
    : '<div class="org-empty">No team annotations yet. Add comments and decisions for risks or recommendations.</div>';

  return `
    <div class="card accent-risk" style="margin-bottom:1rem">
      <div class="card-title"><span class="icon">🗳️</span> Team Annotation Layer</div>
      <div class="form-row-3" style="margin-bottom:.5rem">
        <div class="form-group" style="margin:0">
          <label for="annotation-type">Type</label>
          <select id="annotation-type">
            <option value="general">General</option>
            <option value="risk">Risk</option>
            <option value="recommendation">Recommendation</option>
          </select>
        </div>
        <div class="form-group" style="margin:0">
          <label for="annotation-target">Target</label>
          <select id="annotation-target">
            <option value="general">General sprint context</option>
            ${options}
          </select>
        </div>
        <div class="form-group" style="margin:0">
          <label for="annotation-decision">Decision</label>
          <select id="annotation-decision">
            <option value="monitor">Monitor</option>
            <option value="accept">Accept</option>
            <option value="defer">Defer</option>
            <option value="escalate">Escalate</option>
          </select>
        </div>
      </div>
      <div class="form-row" style="margin-bottom:.5rem">
        <div class="form-group" style="margin:0">
          <label for="annotation-author">Author</label>
          <input id="annotation-author" type="text" placeholder="e.g. PM, Eng Lead" />
        </div>
        <div class="form-group" style="margin:0">
          <label for="annotation-comment">Comment</label>
          <input id="annotation-comment" type="text" placeholder="Decision rationale or follow-up action" />
        </div>
      </div>
      <div class="btn-row" style="margin-bottom:.6rem">
        <button class="btn btn-primary" onclick="saveTeamAnnotation()">Add Decision Log Entry</button>
      </div>
      <div class="annotation-list">${list}</div>
    </div>
  `;
}

function applyDashboardMotionChoreography() {
  const revealNodes = document.querySelectorAll('#dashboard-content .card, #dashboard-content .stat-card');
  revealNodes.forEach((node, index) => {
    node.classList.remove('reveal-stage');
    node.style.setProperty('--reveal-delay', `${Math.min(index * 32, 420)}ms`);
    requestAnimationFrame(() => {
      node.classList.add('reveal-stage');
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// Dashboard page
// ═══════════════════════════════════════════════════════════════

function renderDashboard() {
  const r = AppState.lastResult;

  if (!r) {
    document.getElementById('dashboard-content').innerHTML = `
      ${buildOrganizationRetrieveCard()}
      ${buildSavedViewsPanel()}
      ${buildMultiOrgComparisonBoard()}
      ${buildTimelinePlaybackPanel()}
      ${buildPredictiveAlertsPanel()}
      <div class="card" style="text-align:center;padding:2.2rem">
        <div style="font-size:2rem;margin-bottom:.5rem">🚀</div>
        <div style="font-size:1rem;font-weight:600;margin-bottom:.25rem">No Active Inference Loaded</div>
        <div style="font-size:.82rem;color:var(--text-sec);margin-bottom:1rem">Run or retrieve a sprint inference to populate the executive dashboard cards.</div>
        ${buildShortcutActionsHtml()}
        <div class="btn-row" style="justify-content:center;margin-top:.6rem">
          <button class="btn btn-ghost" onclick="copyDashboardDeepLink()">Copy Shared Link</button>
        </div>
      </div>
    `;
    applyDashboardMotionChoreography();
    syncDeepLinkState();
    refreshFocusableCards();
    return;
  }

  const a = r.sprint_analysis || {};
  const risks = r.identified_risks || [];
  const recs = r.recommendations || [];
  const rm = r.run_metrics || {};

  Object.values(AppState.dashboardCharts).forEach(c => c.destroy && c.destroy());
  AppState.dashboardCharts = {};

  document.getElementById('dashboard-content').innerHTML = `
    ${buildOrganizationRetrieveCard()}
    ${buildSavedViewsPanel()}
    ${buildMultiOrgComparisonBoard()}
    ${buildTimelinePlaybackPanel()}
    ${buildPredictiveAlertsPanel()}
    <div class="grid-auto" style="margin-bottom:1.1rem">
      <div class="stat-card ${colorClass(a.health_score ?? 0)}"><div class="stat-label">Health Score</div><div class="stat-value">${(a.health_score ?? 0).toFixed(1)}</div><div class="stat-sub">${statusLabel(a.health_status)}</div></div>
      <div class="stat-card ${colorClass(a.completion_probability ?? 0)}"><div class="stat-label">Completion</div><div class="stat-value">${(a.completion_probability ?? 0).toFixed(0)}%</div><div class="stat-sub">Prediction</div></div>
      <div class="stat-card cyan"><div class="stat-label">Risks</div><div class="stat-value">${risks.length}</div><div class="stat-sub">Identified</div></div>
      <div class="stat-card violet"><div class="stat-label">Actions</div><div class="stat-value">${recs.length}</div><div class="stat-sub">Recommended</div></div>
      <div class="stat-card green"><div class="stat-label">Latency</div><div class="stat-value">${(rm.latency_seconds ?? 0).toFixed(1)}s</div><div class="stat-sub">Pipeline</div></div>
    </div>
    <div class="grid-2" style="margin-bottom:1rem">
      <div class="card">
        <div class="card-title"><span class="icon">📊</span> Sprint Health Radar</div>
        <div class="chart-container" style="height:250px"><canvas id="dashboard-chart-radar"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title"><span class="icon">⚠️</span> Risk Distribution</div>
        <div class="chart-container" style="height:250px"><canvas id="dashboard-chart-risks"></canvas></div>
      </div>
    </div>
    <div class="card accent-risk">
      <div class="card-title"><span class="icon">💡</span> Top Recommendations</div>
      <div class="item-list">
        ${recs.slice(0, 3).map(rec => {
          const p = rec.priority || 'medium';
          return `<div class="item-card"><div class="item-header"><span class="item-title">${rec.title || 'Recommendation'}</span><span class="badge badge-${p}">${p}</span></div><div class="item-desc">${rec.description || ''}</div></div>`;
        }).join('') || '<div class="item-card"><div class="item-desc">No recommendations yet.</div></div>'}
      </div>
    </div>
    <div class="grid-2" style="margin-top:1rem">
      ${buildExportCenterPanel()}
      ${buildExperimentNotesPanel()}
    </div>
    ${buildEvidenceQualityScorecard()}
    ${buildMetricDrilldownPanel(r)}
    ${buildTeamAnnotationLayer(risks, recs)}
  `;

  // Render dashboard charts
  setTimeout(() => {
    const radarCtx = document.getElementById('dashboard-chart-radar');
    if (radarCtx) {
      AppState.dashboardCharts.radar = new Chart(radarCtx, {
        type: 'radar',
        data: {
          labels: ['Health', 'Delivery', 'Momentum', 'Quality', 'Collaboration', 'Completion'],
          datasets: [{
            data: [a.health_score ?? 0, a.delivery_score ?? 0, a.momentum_score ?? 0, a.quality_score ?? 0, a.collaboration_score ?? 0, a.completion_probability ?? 0],
            borderColor: '#818cf8', backgroundColor: '#818cf820', borderWidth: 2,
            pointBackgroundColor: '#818cf8', pointBorderColor: '#fff', pointBorderWidth: 1, pointRadius: 4,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { r: { beginAtZero: true, max: 100, ticks: { color: '#64748b', backdropColor: 'transparent' }, grid: { color: '#1a254040' }, angleLines: { color: '#1a254040' }, pointLabels: { color: '#94a3b8', font: { size: 11, weight: '600' } } } },
        },
      });
    }
    const dCtx = document.getElementById('dashboard-chart-risks');
    if (dCtx) {
      const high = risks.filter(item => (item.severity ?? 0) >= 0.7).length;
      const med = risks.filter(item => (item.severity ?? 0) >= 0.4 && (item.severity ?? 0) < 0.7).length;
      const low = risks.filter(item => (item.severity ?? 0) < 0.4).length;
      AppState.dashboardCharts.risks = new Chart(dCtx, {
        type: 'doughnut',
        data: {
          labels: ['High', 'Medium', 'Low'],
          datasets: [{ data: [high, med, low], backgroundColor: ['#fb718560', '#fbbf2460', '#34d39960'], borderColor: ['#fb7185', '#fbbf24', '#34d399'], borderWidth: 2 }],
        },
        options: {
          responsive: true, maintainAspectRatio: false, cutout: '65%',
          plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 11 }, padding: 16 } } },
        },
      });
    }
  }, 50);

  applyDashboardMotionChoreography();
  syncDeepLinkState();
  refreshFocusableCards();
}

// ═══════════════════════════════════════════════════════════════
// Accordion toggle
// ═══════════════════════════════════════════════════════════════

function toggleAccordion(id) {
  const toggle = document.querySelector(`[data-acc="${id}"]`);
  const body = document.getElementById(id);
  toggle.classList.toggle('open');
  body.classList.toggle('open');
}

// ═══════════════════════════════════════════════════════════════
// Health check
// ═══════════════════════════════════════════════════════════════

async function checkHealth() {
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    const dot = document.getElementById('health-dot');
    const text = document.getElementById('health-text');
    if (data.orchestrator_ready) {
      dot.style.background = 'var(--emerald)';
      text.textContent = 'System Ready';
    } else {
      dot.style.background = 'var(--amber)';
      text.textContent = 'Orchestrator Unavailable';
    }
  } catch (_) {
    const dot = document.getElementById('health-dot');
    const text = document.getElementById('health-text');
    dot.style.background = 'var(--rose)';
    text.textContent = 'Disconnected';
  }
}

// ═══════════════════════════════════════════════════════════════
// Ollama Model Selection
// ═══════════════════════════════════════════════════════════════

async function loadAvailableModels() {
  try {
    console.log('Fetching available models from /api/models...');
    const resp = await fetch('/api/models');
    console.log('API response status:', resp.status);
    const data = await resp.json();
    console.log('API response data:', data);

    if (data.status === 'ok' && data.models) {
      console.log('Successfully loaded', data.models.length, 'models');
      AppState.models.available = data.models;
      AppState.models.current = data.current_model;
      AppState.models.selected = AppState.models.current;
      console.log('AppState.models updated:', AppState.models);
      populateModelSelector();
    } else {
      console.warn('Failed to load models:', data.error);
      logger.warn('Failed to load models:', data.error);
      showToast('Could not load available models', 'warning');
    }
  } catch (error) {
    console.error('Error fetching models:', error);
    logger.error('Error fetching models:', error);
    showToast('Failed to fetch available models', 'warning');
  }
}

function populateModelSelector() {
  const selector = document.getElementById('model-selector');
  console.log('populateModelSelector called. Selector element:', selector);
  if (!selector) {
    console.warn('Model selector element not found in DOM');
    return;
  }

  selector.innerHTML = '';

  if (AppState.models.available.length === 0) {
    console.warn('No models available to populate');
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No models available';
    option.disabled = true;
    selector.appendChild(option);
    return;
  }

  console.log('Populating selector with', AppState.models.available.length, 'models');
  AppState.models.available.forEach(model => {
    const option = document.createElement('option');
    option.value = model.name;

    // Format model display name
    let displayName = model.name;
    if (model.name === AppState.models.current) {
      displayName += ' (default)';
    }
    option.textContent = displayName;

    // Select the current model by default
    if (model.name === AppState.models.current) {
      option.selected = true;
    }

    selector.appendChild(option);
  });

  console.log('Model selector populated with', selector.options.length - 1, 'options');

  // Add change event listener
  selector.addEventListener('change', (e) => {
    AppState.models.selected = e.target.value || AppState.models.current;
    console.log('Model selected changed to:', AppState.models.selected);
  });
}

// ═══════════════════════════════════════════════════════════════
// Settings
// ═══════════════════════════════════════════════════════════════

function toggleAutoRefresh(checked) {
  AppState.settings.autoRefresh = checked;
  if (checked) {
    showToast('Auto-refresh enabled (30s)', 'info');
  }
}

function onSyntheticStepChange() {
  const step = document.getElementById('ingestion-synthetic-step');
  const syntheticCount = document.getElementById('ingestion-synthetic-count');
  const syntheticPersonas = document.getElementById('ingestion-synthetic-personas');

  if (!step || !syntheticCount || !syntheticPersonas) return;

  const isGenerate = step.value === 'generate';
  syntheticCount.disabled = !isGenerate;
  syntheticPersonas.disabled = !isGenerate;
}

function setIngestionStatus(message, tone = 'info') {
  const statusEl = document.getElementById('ingestion-status');
  if (!statusEl) return;

  statusEl.textContent = message;
  statusEl.classList.remove('success', 'error');
  if (tone === 'success' || tone === 'error') {
    statusEl.classList.add(tone);
  }
}

function setIngestionOutput(text) {
  const outputEl = document.getElementById('ingestion-output');
  if (!outputEl) return;
  outputEl.textContent = text;
  outputEl.scrollTop = 0;
}

function clearIngestionOutput() {
  setIngestionStatus('Ready. Configure organization and repos, then run.', 'info');
  setIngestionOutput('No ingestion run yet.');
}

function parseIngestionRepos(rawRepos) {
  const values = String(rawRepos || '')
    .split(/[\n,]/)
    .map(value => value.trim())
    .filter(Boolean);

  const deduped = [];
  const seen = new Set();

  values.forEach(repo => {
    const key = repo.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    deduped.push(repo);
  });

  return deduped;
}

async function runOrgPipelineIngestion() {
  if (AppState.ingestion.running) return;

  const orgLinkInput = document.getElementById('ingestion-org-link');
  const reposInput = document.getElementById('ingestion-repos');
  const repoCountInput = document.getElementById('ingestion-repo-count');
  const syntheticStepInput = document.getElementById('ingestion-synthetic-step');
  const syntheticCountInput = document.getElementById('ingestion-synthetic-count');
  const syntheticPersonasInput = document.getElementById('ingestion-synthetic-personas');
  const includeForksInput = document.getElementById('ingestion-include-forks');
  const queryTestInput = document.getElementById('ingestion-enable-query-test');
  const dryRunInput = document.getElementById('ingestion-dry-run');
  const runBtn = document.getElementById('ingestion-run-btn');

  if (!orgLinkInput || !reposInput || !repoCountInput || !syntheticStepInput || !syntheticCountInput ||
      !syntheticPersonasInput || !includeForksInput || !queryTestInput || !dryRunInput || !runBtn) {
    showToast('Ingestion controls are not available on this page.', 'error');
    return;
  }

  const orgLink = orgLinkInput.value.trim();
  if (!orgLink) {
    setIngestionStatus('Organization URL or name is required.', 'error');
    showToast('Provide organization URL or name first.', 'error');
    return;
  }

  const repos = parseIngestionRepos(reposInput.value);
  let repoCount = Number.parseInt(repoCountInput.value || '1', 10);
  if (!Number.isFinite(repoCount) || repoCount < 1) {
    repoCount = 1;
  }
  if (repos.length > repoCount) {
    repoCount = repos.length;
    repoCountInput.value = String(repoCount);
  }

  const syntheticStep = syntheticStepInput.value;
  let syntheticCount = Number.parseInt(syntheticCountInput.value || '100', 10);
  if (!Number.isFinite(syntheticCount) || syntheticCount < 1) {
    syntheticCount = 100;
  }

  const payload = {
    org_link: orgLink,
    repos,
    repo_count: repoCount,
    include_forks: includeForksInput.checked,
    synthetic_step: syntheticStep,
    synthetic_count: syntheticCount,
    synthetic_personas: syntheticPersonasInput.value,
    no_query_test: !queryTestInput.checked,
    dry_run: dryRunInput.checked,
  };

  AppState.ingestion.running = true;
  const originalLabel = runBtn.textContent;
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';

  setIngestionStatus('Running org pipeline. This can take a few minutes...', 'info');
  setIngestionOutput('Submitting request...');

  try {
    const response = await fetch('/api/ingestion/org-pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      const message = data.error || 'Org pipeline request failed.';
      setIngestionStatus(message, 'error');
      setIngestionOutput(message);
      showToast(message, 'error');
      return;
    }

    const jobId = data.job_id;
    if (!jobId) {
      const message = 'Org pipeline did not return a job id.';
      setIngestionStatus(message, 'error');
      setIngestionOutput(message);
      showToast(message, 'error');
      return;
    }

    let finalData = null;

    while (AppState.ingestion.running) {
      await new Promise(resolve => setTimeout(resolve, 2000));

      const statusResponse = await fetch(`/api/ingestion/org-pipeline/${encodeURIComponent(jobId)}`);
      const statusData = await statusResponse.json();

      if (!statusResponse.ok) {
        const message = statusData.error || 'Failed to fetch org pipeline status.';
        setIngestionStatus(message, 'error');
        setIngestionOutput(message);
        showToast(message, 'error');
        return;
      }

      const lines = [];
      lines.push(`Job ID: ${statusData.job_id || jobId}`);
      lines.push(`$ ${statusData.command || 'npm run org:pipeline -- ...'}`);
      lines.push('');
      lines.push(`Status: ${statusData.status || 'unknown'}`);

      const durationSeconds = Number(statusData.duration_seconds);
      if (Number.isFinite(durationSeconds)) {
        lines.push(`Duration: ${durationSeconds.toFixed(3)}s`);
      }
      if (statusData.returncode !== undefined && statusData.returncode !== null) {
        lines.push(`Return Code: ${statusData.returncode}`);
      }
      if (statusData.stdout_truncated) {
        lines.push('Note: stdout truncated to latest logs.');
      }
      if (statusData.stderr_truncated) {
        lines.push('Note: stderr truncated to latest logs.');
      }
      lines.push('');
      lines.push('[STDOUT]');
      lines.push(
        statusData.stdout ||
        (statusData.status === 'queued' || statusData.status === 'running'
          ? '(pipeline running; logs will be shown after completion)'
          : '(no stdout)')
      );
      lines.push('');
      lines.push('[STDERR]');
      lines.push(statusData.stderr || '(no stderr)');
      setIngestionOutput(lines.join('\n'));

      if (statusData.status === 'queued') {
        setIngestionStatus('Org pipeline is queued...', 'info');
        continue;
      }

      if (statusData.status === 'running') {
        setIngestionStatus('Org pipeline is running. Full history ingestion can take several minutes.', 'info');
        continue;
      }

      finalData = statusData;
      break;
    }

    if (!finalData) {
      return;
    }

    if (finalData.status === 'success') {
      setIngestionStatus('Org pipeline completed successfully.', 'success');
      showToast('Org pipeline completed successfully.', 'success');
      await loadSprintFiles();
    } else if (finalData.status === 'timeout') {
      const timeoutMessage = finalData.error || 'Org pipeline timed out.';
      setIngestionStatus(timeoutMessage, 'error');
      showToast(timeoutMessage, 'error');
    } else {
      setIngestionStatus('Org pipeline finished with errors. Review logs below.', 'error');
      showToast('Org pipeline finished with errors.', 'error');
    }
  } catch (error) {
    const message = `Org pipeline request failed: ${error.message}`;
    setIngestionStatus(message, 'error');
    setIngestionOutput(message);
    showToast(message, 'error');
  } finally {
    AppState.ingestion.running = false;
    runBtn.disabled = false;
    runBtn.textContent = originalLabel;
  }
}

// ═══════════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  loadPersistedViewState();
  const restoredState = applyDeepLinkStateFromUrl();
  if (!restoredState) {
    AppState.currentPage = 'analyze';
  }

  setInputMode(AppState.inputMode);
  navigateTo(AppState.currentPage || 'analyze');
  loadSprintFiles();
  await loadAvailableModels();

  await refreshOrganizationIndex();
  if (AppState.deepLink.pendingRunId && AppState.selectedOrganization) {
    await retrieveOrganizationResult(AppState.deepLink.pendingRunId, { navigate: false, silent: true });
    AppState.deepLink.pendingRunId = '';
  }

  AppState.deepLink.isApplying = false;
  syncDeepLinkState();

  checkHealth();
  onSyntheticStepChange();
  bindSidebarKeyboardNavigation();
  document.addEventListener('keydown', handleGlobalKeyboardShortcuts);
  setInterval(checkHealth, 30000);
});
