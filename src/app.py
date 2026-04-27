import os
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"

from flask import Flask, request, jsonify
import torch
from classifier import load_classifier, classify_image, get_class_names
from rag import load_rag, _retrieve, _answer
import json

app = Flask(__name__)

WEIGHTS = "../models/resnet_plants.pth"
SL_PATH = "../data/class_names.json"
KB_PATH = "../data/wiki_kb.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading models...")
rag = load_rag(KB_PATH, device)

with open(SL_PATH) as f:
    class_names = json.load(f)

classifier = load_classifier(WEIGHTS, len(class_names), device)
print("ready on http://localhost:5000")


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Duke Gardens Index</title>
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #0a0e08;
    --panel:   #111408;
    --border:  #1e2a14;
    --green:   #c8a951;
    --muted:   #4a5a3a;
    --text:    #e8e0cc;
    --sub:     #8a9070;
    }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    min-height: 100vh;
    display: grid;
    grid-template-columns: 420px 1fr;
    grid-template-rows: auto 1fr;
  }

header {
  grid-column: 1 / -1;
  padding: 24px 32px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: baseline;
  gap: 16px;
  background: linear-gradient(90deg, #0d1a08 0%, #0a0e08 60%);
}

header h1 {
  font-family: 'Libre Baskerville', serif;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--green);
  text-transform: uppercase;
}

header span {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

  /* LEFT PANEL */
  .left {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  /* Drop zone */
  #dropzone {
    margin: 24px;
    border: 1px dashed var(--border);
    border-radius: 4px;
    height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
  }

  #dropzone.over {
    border-color: var(--green);
    background: rgba(184,217,141,0.04);
  }

  #dropzone img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0.85;
  }

  #dropzone .label {
    color: var(--muted);
    font-size: 11px;
    text-align: center;
    line-height: 1.8;
    pointer-events: none;
    position: relative;
    z-index: 1;
  }

  #dropzone .icon {
    font-size: 28px;
    opacity: 0.35;
  }

  #file-input { display: none; }

  /* Result box */
  #result {
    margin: 0 24px 24px;
    padding: 14px 16px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    min-height: 56px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  #result .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: background 0.3s;
  }

  #result.identified .dot { background: var(--green); }

  #result .species {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 15px;
    color: var(--green);
  }

  #result .placeholder { color: var(--muted); }

  /* Spinner */
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid var(--border);
    border-top-color: var(--green);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* RIGHT PANEL — chat */
  .right {
    display: flex;
    flex-direction: column;
  }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-width: 680px;
    animation: fadeup 0.2s ease;
  }

  @keyframes fadeup {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .msg .who {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.user .who { color: var(--green); }

  .msg .bubble {
    padding: 10px 14px;
    border-radius: 3px;
    line-height: 1.6;
    background: var(--panel);
    border: 1px solid var(--border);
  }

  .msg.user .bubble {
    background: rgba(184,217,141,0.07);
    border-color: rgba(184,217,141,0.2);
  }

  .empty-state {
    margin: auto;
    text-align: center;
    color: var(--muted);
    line-height: 2;
  }

  .empty-state .big {
    font-family: 'Playfair Display', serif;
    font-size: 18px;
    font-style: italic;
    color: var(--sub);
    display: block;
    margin-bottom: 6px;
  }

  /* Input row */
  #input-row {
    padding: 16px 28px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
  }

  #query {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    padding: 10px 14px;
    outline: none;
    transition: border-color 0.2s;
  }

  #query:focus { border-color: var(--green); }
  #query::placeholder { color: var(--muted); }

  #send-btn {
    background: var(--green);
    color: var(--bg);
    border: none;
    border-radius: 3px;
    padding: 10px 18px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: opacity 0.15s;
    letter-spacing: 0.04em;
  }

  #send-btn:hover { opacity: 0.85; }
  #send-btn:disabled { opacity: 0.35; cursor: default; }
</style>
</head>
<body>

<header>
  <h1>DGI</h1>
  <span>Duke Gardens Index</span>
</header>

<div class="left">
  <div id="dropzone">
    <span class="icon">⬡</span>
    <span class="label">drop an image here<br>or click to browse</span>
    <input type="file" id="file-input" accept="image/*">
  </div>

  <div id="result">
    <div class="dot"></div>
    <span class="placeholder">awaiting image</span>
  </div>
</div>

<div class="right">
  <div id="messages">
    <div class="empty-state">
      <span class="big">ask anything</span>
      drop a plant image, then chat about it
    </div>
  </div>
  <div id="input-row">
    <input id="query" type="text" placeholder="what environment does it grow in?" autocomplete="off">
    <button id="send-btn">send</button>
  </div>
</div>

<script>
  let currentSpecies = null;

  const dropzone  = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const result    = document.getElementById('result');
  const messages  = document.getElementById('messages');
  const queryEl   = document.getElementById('query');
  const sendBtn   = document.getElementById('send-btn');

  // --- drag & drop ---
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('over');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImage(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleImage(fileInput.files[0]);
  });

  function handleImage(file) {
    // show preview
    const reader = new FileReader();
    reader.onload = e => {
      dropzone.querySelectorAll('img, .label, .icon').forEach(el => el.remove());
      const img = document.createElement('img');
      img.src = e.target.result;
      dropzone.appendChild(img);
    };
    reader.readAsDataURL(file);

    // show spinner
    result.classList.remove('identified');
    result.innerHTML = '<div class="spinner"></div><span style="color:var(--muted)">identifying...</span>';

    const formData = new FormData();
    formData.append('image', file);

    fetch('/classify', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        currentSpecies = data.species;
        result.classList.add('identified');
        result.innerHTML = `<div class="dot"></div><span class="species">${data.species}</span>`;
        addMessage('bot', `identified as <em>${data.species}</em>. ask me anything about it.`);
      })
      .catch(() => {
        result.innerHTML = '<div class="dot"></div><span style="color:#c87;"> error classifying</span>';
      });
  }

  // --- chat ---
  sendBtn.addEventListener('click', sendQuery);
  queryEl.addEventListener('keydown', e => { if (e.key === 'Enter') sendQuery(); });

  function sendQuery() {
    const q = queryEl.value.trim();
    if (!q) return;

    addMessage('user', q);
    queryEl.value = '';
    sendBtn.disabled = true;

    // remove empty state if present
    const empty = messages.querySelector('.empty-state');
    if (empty) empty.remove();

    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, species: currentSpecies || '' })
    })
      .then(r => r.json())
      .then(data => {
        addMessage('bot', data.answer);
        sendBtn.disabled = false;
      })
      .catch(() => {
        addMessage('bot', 'error — check server logs');
        sendBtn.disabled = false;
      });
  }

  function addMessage(who, text) {
    const empty = messages.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = `msg ${who}`;
    div.innerHTML = `<span class="who">${who === 'user' ? 'you' : 'bot'}</span>
                     <div class="bubble">${text}</div>`;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML


@app.route("/classify", methods=["POST"])
def classify():
    f = request.files["image"]
    f.save("/tmp/upload.jpg")
    species = classify_image("/tmp/upload.jpg", classifier, class_names, device)
    return jsonify({"species": species})


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.json
    query   = data["query"]
    species = data.get("species", "")
    full_q = f"{species} {query}" if species else query
    context = _retrieve(full_q, rag, device)
    ans     = _answer(query, context, rag, device)
    return jsonify({"answer": ans})


if __name__ == "__main__":
    app.run(port=5000, debug=False)