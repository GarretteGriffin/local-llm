let sessionId = null;

const chatEl = document.getElementById('chat');
const messageEl = document.getElementById('message');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');
const attachmentsEl = document.getElementById('attachments');
const toolStatusEl = document.getElementById('tool-status');
const toolTextEl = document.querySelector('.tool-text');

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
        `<li><a href="${s.href}" target="_blank" rel="noopener noreferrer">${s.label}</a></li>`,
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
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
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
});

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

  addMessage('user', text || '(files attached)');
  const botDiv = addMessage('bot');
  const botContentEl = botDiv.querySelector('.content');
  const botSourcesEl = botDiv.querySelector('.sources');

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
            break;
          case 'tool':
            if (payload.status === 'running') {
              showTool(payload.tool, payload.message);
            } else {
              hideTool();
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
              renderSources(botSourcesEl, extractSources(botContentEl));
            }
            break;
          case 'content':
            if (!payload.content) break;
            fullContent += payload.content;
            {
              const displayContent = normalizeMarkdownLinks(cleanText(fullContent));
              botContentEl.innerHTML = marked.parse(displayContent);
              linkifyElement(botContentEl);
              setAnchorBehavior(botContentEl);
              renderSources(botSourcesEl, extractSources(botContentEl));
            }
            break;
          default:
            break;
        }

        chatEl.scrollTop = chatEl.scrollHeight;
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
