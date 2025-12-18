async function api(path, opts = {}) {
  const headers = opts.headers || {};

  // CSRF token is set on successful login (admin_auth) as a non-HttpOnly cookie.
  const csrf = document.cookie
    .split(';')
    .map(s => s.trim())
    .find(s => s.startsWith('admin_csrf='));

  if (csrf && (opts.method && opts.method !== 'GET')) {
    headers['X-CSRF-Token'] = decodeURIComponent(csrf.split('=')[1] || '');
  }

  const res = await fetch(path, {
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
  });

  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }

  if (!res.ok) {
    const msg = data && (data.detail || data.error) ? (data.detail || data.error) : `HTTP ${res.status}`;
    throw new Error(msg);
  }

  return data;
}

function setStatus(msg, kind = 'info') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.dataset.kind = kind;
}

function mkInput(type, value = '') {
  const i = document.createElement('input');
  i.type = type;
  i.value = value;
  return i;
}

function buildModelsGrid(baseModels, overrideModels) {
  const grid = document.getElementById('modelsGrid');
  grid.innerHTML = '';

  const tiers = Object.keys(baseModels);

  const header = document.createElement('div');
  header.className = 'model-row hint';
  header.innerHTML = `
    <div></div>
    <div>Model name</div>
    <div>Temp</div>
    <div>Max tokens</div>
    <div>Top P</div>
    <div>Top K</div>
    <div>Ctx</div>
  `;
  grid.appendChild(header);

  for (const tier of tiers) {
    const base = baseModels[tier] || {};
    const ovr = (overrideModels && overrideModels[tier]) || {};

    const row = document.createElement('div');
    row.className = 'model-row';

    const tierEl = document.createElement('div');
    tierEl.className = 'tier';
    tierEl.textContent = tier;

    const name = mkInput('text', ovr.name ?? '');
    name.placeholder = base.name || '';

    const temp = mkInput('number', ovr.temperature ?? '');
    temp.placeholder = String(base.temperature ?? '');
    temp.step = '0.05';

    const maxTokens = mkInput('number', ovr.max_tokens ?? '');
    maxTokens.placeholder = String(base.max_tokens ?? '');

    const topP = mkInput('number', ovr.top_p ?? '');
    topP.placeholder = String(base.top_p ?? '');
    topP.step = '0.01';

    const topK = mkInput('number', ovr.top_k ?? '');
    topK.placeholder = String(base.top_k ?? '');

    const ctx = mkInput('number', ovr.context_window ?? '');
    ctx.placeholder = String(base.context_window ?? '');

    row.appendChild(tierEl);
    row.appendChild(name);
    row.appendChild(temp);
    row.appendChild(maxTokens);
    row.appendChild(topP);
    row.appendChild(topK);
    row.appendChild(ctx);

    row.dataset.tier = tier;

    grid.appendChild(row);
  }
}

function collectOverridesFromUI(base) {
  const assistantPrompt = document.getElementById('assistantPrompt').value;
  const analystPrompt = document.getElementById('analystPrompt').value;

  const models = {};
  for (const row of document.querySelectorAll('.model-row')) {
    const tier = row.dataset.tier;
    if (!tier) continue;

    const inputs = row.querySelectorAll('input');
    const [name, temp, maxTokens, topP, topK, ctx] = inputs;

    const tierOverride = {};
    if (name.value.trim()) tierOverride.name = name.value.trim();
    if (temp.value !== '') tierOverride.temperature = Number(temp.value);
    if (maxTokens.value !== '') tierOverride.max_tokens = Number(maxTokens.value);
    if (topP.value !== '') tierOverride.top_p = Number(topP.value);
    if (topK.value !== '') tierOverride.top_k = Number(topK.value);
    if (ctx.value !== '') tierOverride.context_window = Number(ctx.value);

    if (Object.keys(tierOverride).length) models[tier] = tierOverride;
  }

  const tools = {
    web_search_enabled: Boolean(document.getElementById('webSearchEnabled').checked),
  };

  const system_prompts = {
    assistant: assistantPrompt,
    analyst: analystPrompt,
  };

  return { system_prompts, models, tools };
}

async function load() {
  setStatus('Loading…');
  const data = await api('/api/config');

  const overrides = data.overrides || {};

  document.getElementById('assistantPrompt').value = (overrides.system_prompts && overrides.system_prompts.assistant) || '';
  document.getElementById('analystPrompt').value = (overrides.system_prompts && overrides.system_prompts.analyst) || '';

  const baseModels = (data.base && data.base.models) || {};
  const overrideModels = overrides.models || {};

  buildModelsGrid(baseModels, overrideModels);

  const toolOverrides = overrides.tools || {};
  const webSearchEnabled = (typeof toolOverrides.web_search_enabled === 'boolean')
    ? toolOverrides.web_search_enabled
    : true;
  document.getElementById('webSearchEnabled').checked = webSearchEnabled;

  setStatus('Loaded');
  window.__adminBase = data.base;
}

async function save() {
  setStatus('Saving…');

  const payload = collectOverridesFromUI(window.__adminBase || {});
  await api('/api/config', { method: 'PUT', body: JSON.stringify(payload) });

  setStatus('Saved');
}

async function reset() {
  if (!confirm('Reset all overrides? This will revert the chat service to base defaults.')) return;
  setStatus('Resetting…');
  await api('/api/reset', { method: 'POST', body: '{}' });
  await load();
  setStatus('Reset complete');
}

document.getElementById('refreshBtn').addEventListener('click', () => load().catch(e => setStatus(e.message, 'error')));
document.getElementById('saveBtn').addEventListener('click', () => save().catch(e => setStatus(e.message, 'error')));
document.getElementById('resetBtn').addEventListener('click', () => reset().catch(e => setStatus(e.message, 'error')));

load().catch(e => setStatus(e.message, 'error'));
