import re
import json
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "microsoft/phi-2"

os.environ["TRANSFORMERS_CACHE"] = "./model_cache"


def load_rag(kb_path, device):
    with open(kb_path) as f:
        kb = json.load(f)
    contexts = [f"Species: {item['species']}\n{item['text']}" for item in kb]

    print("loading embedding model...")
    embed_tok   = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL).to(device)

    print("computing embeddings...")
    embeddings = _embed(contexts, embed_model, embed_tok, device)

    llm, llm_tok = None, None
    if device.type == "cuda":
        print("gpu detected, loading phi-2...")
        config = AutoConfig.from_pretrained(LLM_MODEL)
        config.pad_token_id = config.eos_token_id
        llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm_tok.pad_token = llm_tok.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, config=config, torch_dtype=torch.float32, trust_remote_code=True
        ).to(device)
    else:
        print("cpu mode — using retrieval only")

    return {"contexts": contexts, "embeddings": embeddings,
            "embed_model": embed_model, "embed_tok": embed_tok,
            "llm": llm, "llm_tok": llm_tok, "kb": kb}


def chat(species, rag, device):
    print(f"\nchatting about: {species or 'any plant'} — type 'quit' to exit\n")
    while True:
        query = input("you: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        context = _retrieve(f"{species} {query}" if species else query, rag, device)
        print(f"bot: {_answer(query, context, rag, device)}\n")


# --- internal ---

def _embed(texts, model, tokenizer, device):
    vecs = []
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs)
        vecs.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(vecs)


def _retrieve(query, rag, device):
    for ctx, item in zip(rag["contexts"], rag["kb"]):
        if item["species"].lower() in query.lower():
            return ctx

    # fallback: semantic search
    q    = _embed([query], rag["embed_model"], rag["embed_tok"], device)
    sims = np.dot(rag["embeddings"], q.T).squeeze()
    sims /= np.linalg.norm(rag["embeddings"], axis=1) * np.linalg.norm(q)
    return rag["contexts"][np.argmax(sims)]


def _answer(query, context, rag, device):
    if rag["llm"] is not None:
        prompt = (f"Answer the question using the information below.\n"
                  f"Context:\n{context}\nQuestion: {query}\nAnswer:")
        tok    = rag["llm_tok"]
        inputs = tok(prompt, return_tensors="pt").to(device)
        plen   = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = rag["llm"].generate(**inputs, max_new_tokens=150,
                                      pad_token_id=tok.eos_token_id, do_sample=False)
        raw  = tok.decode(out[0][plen:], skip_special_tokens=True)
        stop = re.search(r"(Exercise \d|Question:|Context:)", raw)
        return (raw[:stop.start()] if stop else raw).split("\n")[0].strip()
    else:
        for item in rag["kb"]:
            if item["species"].lower() in context.lower():
                return f"[running on CPU — local LLM unavailable] {item['text']}"
        return context