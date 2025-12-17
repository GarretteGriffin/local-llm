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
    div.innerHTML = text ? marked.parse(text) : '';
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
            botDiv.innerHTML = marked.parse(fullContent);
            break;
          case 'content':
            if (!payload.content) break;
            fullContent += payload.content;
            const displayContent = cleanText(fullContent);
            botDiv.innerHTML = marked.parse(displayContent);
            botDiv.querySelectorAll('a').forEach((a) => (a.target = '_blank'));
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
