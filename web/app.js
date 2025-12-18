let sessionId = null;
let activeConversationId = null;

const chatEl = document.getElementById('chat');
const messageEl = document.getElementById('message');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const attachmentsEl = document.getElementById('attachments');
const toolStatusEl = document.getElementById('tool-status');
const toolTextEl = document.querySelector('.tool-text');
const convoListEl = document.getElementById('convoList');
const newChatBtn = document.getElementById('newChatBtn');
const activityListEl = document.getElementById('activityList');
const clearActivityBtn = document.getElementById('clearActivityBtn');

const STORAGE_KEY = 'local-llm-conversations-v1';

function nowIso() {
  return new Date().toISOString();
}

function makeId() {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return `c_${Math.random().toString(16).slice(2)}_${Date.now()}`;
}

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const convos = raw ? JSON.parse(raw) : [];
    return Array.isArray(convos) ? convos : [];
  } catch {
    return [];
  }
}

function saveConversations(conversations) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations || []));
  } catch {
    // ignore
  }
}

function getConversation(conversationId) {
  return loadConversations().find((c) => c.id === conversationId) || null;
}

function upsertConversation(conversation) {
  const conversations = loadConversations();
  const idx = conversations.findIndex((c) => c.id === conversation.id);
  if (idx >= 0) conversations[idx] = conversation;
  else conversations.unshift(conversation);
  saveConversations(conversations);
}

function deleteConversation(conversationId) {
  const conversations = loadConversations().filter((c) => c.id !== conversationId);
  saveConversations(conversations);
}

function setActiveConversation(conversationId) {
  activeConversationId = conversationId;
  const convo = getConversation(conversationId);
  sessionId = convo ? convo.sessionId || null : null;
  renderConversationList();
  renderConversation(convo);
}

function ensureActiveConversation(seedTitle) {
  if (activeConversationId) return getConversation(activeConversationId);

  const conversation = {
    id: makeId(),
    title: (seedTitle || 'New chat').slice(0, 80),
    sessionId: null,
    createdAt: nowIso(),
    updatedAt: nowIso(),
    messages: [],
  };
  upsertConversation(conversation);
  setActiveConversation(conversation.id);
  return conversation;
}

function summarizeTitle(text) {
  const t = String(text || '').trim();
  if (!t) return 'New chat';
  return t.length > 48 ? `${t.slice(0, 48)}…` : t;
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function normalizeMarkdownLinks(text) {
  // Fix common streaming/newline breaks that split markdown link syntax.
  return String(text || '').replace(/\]\s*\n\s*\(/g, '](');
}

function setAnchorBehavior(root) {
  root.querySelectorAll('a[href]').forEach((a) => {
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
  });
}

function linkifyElement(root) {
  // Convert bare URLs in rendered HTML into clickable links.
  // Skips existing links and code blocks.
  const urlRe = /(https?:\/\/[^\s<]+|www\.[^\s<]+)/gi;
  const urlTestRe = /(https?:\/\/[^\s<]+|www\.[^\s<]+)/i;

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (parent.closest('a, code, pre, script, style')) return NodeFilter.FILTER_REJECT;
      if (!node.nodeValue || !urlTestRe.test(node.nodeValue)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  for (const textNode of nodes) {
    const text = textNode.nodeValue || '';
    const frag = document.createDocumentFragment();
    let lastIndex = 0;

    // Reset lastIndex because urlRe is global.
    urlRe.lastIndex = 0;
    let match;
    while ((match = urlRe.exec(text)) !== null) {
      const start = match.index;
      const raw = match[0];

      if (start > lastIndex) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex, start)));
      }

      // Trim trailing punctuation that often follows URLs in prose.
      let display = raw;
      let url = raw;
      let trailing = '';
      while (/[\]\)\}>,.;:!?]$/.test(url)) {
        trailing = url.slice(-1) + trailing;
        url = url.slice(0, -1);
        display = display.slice(0, -1);
      }

      const a = document.createElement('a');
      a.textContent = display;
      a.href = url.startsWith('www.') ? `https://${url}` : url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      frag.appendChild(a);

      if (trailing) frag.appendChild(document.createTextNode(trailing));
      lastIndex = start + raw.length;
    }

    if (lastIndex < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    textNode.parentNode.replaceChild(frag, textNode);
  }
}

function extractSources(root) {
  const seen = new Set();
  const sources = [];

  root.querySelectorAll('a[href]').forEach((a) => {
    const href = a.getAttribute('href') || '';
    if (!/^https?:\/\//i.test(href)) return;
    if (seen.has(href)) return;
    seen.add(href);

    let label = (a.textContent || '').trim();
    if (!label || label === href) {
      try {
        label = new URL(href).hostname;
      } catch {
        label = href;
      }
    }
    sources.push({ href, label });
  });

  return sources;
}

function renderSources(detailsEl, sources) {
  if (!detailsEl) return;
  if (!sources.length) {
    detailsEl.classList.add('hidden');
    detailsEl.open = false;
    detailsEl.innerHTML = '';
    return;
  }

  detailsEl.classList.remove('hidden');
  const items = sources
    .map(
      (s) =>
        s.href
          ? `<li><a href="${s.href}" target="_blank" rel="noopener noreferrer">${s.label}</a></li>`
          : `<li><span>${s.label}</span></li>`,
    )
    .join('');
  detailsEl.innerHTML = `<summary>Sources</summary><ol>${items}</ol>`;
}

messageEl.addEventListener('input', function () {
  this.style.height = 'auto';
  this.style.height = `${this.scrollHeight}px`;
});

function showTool(name, message) {
  if (!toolStatusEl) return;
  toolTextEl.textContent = message || `Running ${name}...`;
  toolStatusEl.classList.remove('hidden');
}

function hideTool() {
  if (!toolStatusEl) return;
  toolStatusEl.classList.add('hidden');
}

function addMessage(role, text = '') {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  if (role === 'bot') {
    div.innerHTML = `<div class="content"></div><details class="sources hidden"></details>`;
    const contentEl = div.querySelector('.content');
    const normalized = normalizeMarkdownLinks(text);
    contentEl.innerHTML = normalized ? marked.parse(normalized) : '';
    linkifyElement(contentEl);
    setAnchorBehavior(contentEl);
    renderSources(div.querySelector('.sources'), extractSources(contentEl));
  } else {
    div.textContent = text;
  }
  chatEl.appendChild(div);
  scrollToBottom();
  return div;
}

function renderConversation(conversation) {
  chatEl.innerHTML = '';
  if (!conversation || !Array.isArray(conversation.messages)) {
    return;
  }
  for (const m of conversation.messages) {
    addMessage(m.role, m.text || '');
  }
  scrollToBottom();
}

function renderConversationList() {
  if (!convoListEl) return;
  const conversations = loadConversations();
  convoListEl.innerHTML = '';

  for (const c of conversations) {
    const item = document.createElement('div');
    item.className = `convo-item${c.id === activeConversationId ? ' active' : ''}`;
    item.setAttribute('role', 'listitem');

    const title = document.createElement('div');
    title.className = 'convo-title';
    title.textContent = c.title || 'New chat';

    const del = document.createElement('button');
    del.className = 'convo-delete';
    del.type = 'button';
    del.textContent = '×';
    del.setAttribute('aria-label', 'Delete conversation');

    item.addEventListener('click', () => setActiveConversation(c.id));
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      const wasActive = c.id === activeConversationId;
      deleteConversation(c.id);
      if (wasActive) {
        activeConversationId = null;
        sessionId = null;
        chatEl.innerHTML = '';
      }
      const remaining = loadConversations();
      if (wasActive && remaining.length) {
        setActiveConversation(remaining[0].id);
      } else {
        renderConversationList();
      }
    });

    item.appendChild(title);
    item.appendChild(del);
    convoListEl.appendChild(item);
  }
}

function appendActivity(entry) {
  if (!activityListEl) return;
  const div = document.createElement('div');
  div.className = 'activity-item';

  const top = document.createElement('div');
  top.className = 'activity-top';

  const name = document.createElement('div');
  name.className = 'activity-name';
  name.textContent = entry.name || 'activity';

  const status = document.createElement('div');
  status.className = 'activity-status';
  status.textContent = entry.status || '';

  const msg = document.createElement('div');
  msg.className = 'activity-msg';
  msg.textContent = entry.message || '';

  top.appendChild(name);
  top.appendChild(status);
  div.appendChild(top);
  if (entry.message) div.appendChild(msg);

  activityListEl.prepend(div);

  // Keep the list from growing unbounded.
  const max = 50;
  while (activityListEl.childElementCount > max) {
    activityListEl.removeChild(activityListEl.lastElementChild);
  }
}

function renderAttachments() {
  const files = Array.from(fileInput.files || []);
  attachmentsEl.textContent = files.length
    ? `Attached: ${files.map((f) => f.name).join(', ')}`
    : '';
}

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', renderAttachments);

clearBtn.addEventListener('click', () => {
  chatEl.innerHTML = '';
  fileInput.value = '';
  renderAttachments();
  hideTool();

  if (activeConversationId) {
    const convo = getConversation(activeConversationId);
    if (convo) {
      convo.messages = [];
      convo.updatedAt = nowIso();
      upsertConversation(convo);
    }
  }
});

if (newChatBtn) {
  newChatBtn.addEventListener('click', () => {
    activeConversationId = null;
    sessionId = null;
    chatEl.innerHTML = '';
    messageEl.focus();
    renderConversationList();
  });
}

if (clearActivityBtn) {
  clearActivityBtn.addEventListener('click', () => {
    if (activityListEl) activityListEl.innerHTML = '';
  });
}


function cleanText(text) {
  return text
    .replace(/\s+\./g, '.')
    .replace(/\s+,/g, ',')
    .replace(/\s+!/g, '!')
    .replace(/\s+\?/g, '?')
    .replace(/\s+:/g, ': ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

async function send() {
  const text = (messageEl.value || '').trim();
  const files = Array.from(fileInput.files || []);
  if (!text && files.length === 0) return;

  const convo = ensureActiveConversation(summarizeTitle(text || '(files attached)'));
  if (!convo.title || convo.title === 'New chat') {
    convo.title = summarizeTitle(text || '(files attached)');
  }
  convo.updatedAt = nowIso();
  convo.messages.push({ role: 'user', text: text || '(files attached)', ts: nowIso() });
  upsertConversation(convo);
  renderConversationList();

  addMessage('user', text || '(files attached)');
  const botDiv = addMessage('bot');
  const botContentEl = botDiv.querySelector('.content');
  const botSourcesEl = botDiv.querySelector('.sources');
  const structuredSources = [];
  const structuredSourceKeys = new Set();
  let hasStructuredSources = false;

  let botMessageIndex = null;
  {
    const current = getConversation(activeConversationId);
    if (current) {
      current.messages.push({ role: 'bot', text: '', ts: nowIso() });
      botMessageIndex = current.messages.length - 1;
      current.updatedAt = nowIso();
      upsertConversation(current);
    }
  }

  messageEl.value = '';
  messageEl.style.height = 'auto';

  const form = new FormData();
  form.append('message', text);
  if (sessionId) form.append('session_id', sessionId);
  for (const file of files) form.append('files', file);

  fileInput.value = '';
  renderAttachments();

  try {
    const resp = await fetch('/chat/stream', { method: 'POST', body: form });
    if (!resp.ok || !resp.body) {
      throw new Error(`Server error: ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let fullContent = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const lines = rawEvent.split('\n');
        let data = '';
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const chunk = line.slice(5);
          data += chunk.startsWith(' ') ? chunk.slice(1) : chunk;
        }
        if (!data.trim()) continue;

        let payload;
        try {
          payload = JSON.parse(data);
        } catch (err) {
          console.error('Invalid SSE payload', err, data);
          continue;
        }

        switch (payload.type) {
          case 'session':
            sessionId = payload.session_id;
            {
              const current = getConversation(activeConversationId);
              if (current && !current.sessionId) {
                current.sessionId = sessionId;
                current.updatedAt = nowIso();
                upsertConversation(current);
                renderConversationList();
              }
            }
            break;
          case 'tool':
            if (payload.status === 'running') {
              showTool(payload.tool, payload.message);
              appendActivity({
                name: payload.tool || 'tool',
                status: 'running',
                message: payload.message || '',
              });
            } else {
              hideTool();
              appendActivity({ name: payload.tool || 'tool', status: 'complete', message: '' });
            }
            break;
          case 'routing':
            break;
          case 'error':
            fullContent += `\n\n*Error: ${payload.message || 'Unknown error'}*`;
            {
              const displayContent = normalizeMarkdownLinks(cleanText(fullContent));
              botContentEl.innerHTML = marked.parse(displayContent);
              linkifyElement(botContentEl);
              setAnchorBehavior(botContentEl);
              renderSources(
                botSourcesEl,
                hasStructuredSources ? structuredSources : extractSources(botContentEl),
              );
            }

            {
              const current = getConversation(activeConversationId);
              if (current && botMessageIndex != null && current.messages[botMessageIndex]) {
                current.messages[botMessageIndex].text = fullContent;
                current.updatedAt = nowIso();
                upsertConversation(current);
              }
            }
            break;
          case 'sources': {
            const incoming = Array.isArray(payload.sources) ? payload.sources : [];
            for (const s of incoming) {
              if (!s) continue;
              const href = typeof s.href === 'string' ? s.href : '';
              const label = (typeof s.label === 'string' ? s.label : '').trim();
              const key = `${href}|${label}`;
              if (!label) continue;
              if (structuredSourceKeys.has(key)) continue;
              structuredSourceKeys.add(key);
              structuredSources.push({ href, label });
            }
            hasStructuredSources = structuredSources.length > 0;
            renderSources(botSourcesEl, structuredSources);
            appendActivity({
              name: 'sources',
              status: 'updated',
              message: `${structuredSources.length} source(s)`,
            });
            break;
          }
          case 'content':
            if (!payload.content) break;
            fullContent += payload.content;
            {
              const displayContent = normalizeMarkdownLinks(cleanText(fullContent));
              botContentEl.innerHTML = marked.parse(displayContent);
              linkifyElement(botContentEl);
              setAnchorBehavior(botContentEl);
              renderSources(
                botSourcesEl,
                hasStructuredSources ? structuredSources : extractSources(botContentEl),
              );
            }

            {
              const current = getConversation(activeConversationId);
              if (current && botMessageIndex != null && current.messages[botMessageIndex]) {
                current.messages[botMessageIndex].text = fullContent;
                current.updatedAt = nowIso();
                upsertConversation(current);
              }
            }
            break;
          default:
            break;
        }

        scrollToBottom();
      }
    }
  } catch (err) {
    botDiv.textContent = `Error: ${err.message}`;
  } finally {
    hideTool();
  }
}

sendBtn.addEventListener('click', send);
messageEl.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    send();
  }
});

// Initial render
renderConversationList();
{
  const conversations = loadConversations();
  if (conversations.length) {
    setActiveConversation(conversations[0].id);
  }
}
