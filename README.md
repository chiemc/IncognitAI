# AI-Powered Private Search Engine

A self-hosted, Perplexity-style search engine. You ask a question, or upload an image. The app searches the web, reads the result pages, stores them in a vector database for retrieval-augmented generation (RAG), and writes an answer with a local LLM. All the models run on your own machine.

Based on the Hugging Face tutorial [*Building an AI-powered search engine from scratch*](https://huggingface.co/blog/as-cle-bert/build-an-ai-powered-search-engine-from-scratch) ([AstraBert/PrAIvateSearch](https://github.com/AstraBert/PrAIvateSearch)). The tutorial targets an older library stack and a CUDA GPU, so getting it running on an Apple Silicon Mac with current libraries took the debugging written up in [DEBUGGING.md](DEBUGGING.md).

## Architecture

```
            ┌──────────────┐
 query ───▶ │  Gradio UI   │  app.py
 (+ image)  └──────┬───────┘
                   │ image? ──▶ Florence-2 caption ──▶ RAKE keywords      image_gen.py
                   ▼
        ┌─────────────────────┐
        │  DuckDuckGo search  │ ──▶ fetch pages ──▶ trafilatura text extraction websearching.py
        └─────────┬───────────┘
                  │
                  ├──▶ chunk + embed (LaBSE) ──▶ Qdrant (semantic memory)
                  │                                  │
                  │  RAG: top-5 chunks for the question ◀┘ (numbered sources)  rag.py
                  ▼
        ┌─────────────────────┐      ┌──────────────────────────┐
        │ Qwen2.5-3B-Instruct │ ◀──▶ │ Postgres (chat history)  │           text_inference.py, memory.py
        └─────────┬───────────┘      └──────────────────────────┘
                  ▼
               answer
```

| Component | Technology |
|---|---|
| UI | Gradio |
| Web search | `ddgs` (DuckDuckGo) |
| Page text extraction | trafilatura + requests |
| Keyword extraction (image captions) | rake-nltk |
| Embeddings | `sentence-transformers/LaBSE` |
| Vector DB | Qdrant (Docker) |
| Conversation memory | PostgreSQL 17 (Docker) + SQLAlchemy |
| LLM | `Qwen/Qwen2.5-3B-Instruct` (transformers; 4-bit on CUDA, fp16 on Apple MPS) |
| Image captioning | Florence-2-large (native transformers implementation) |

## Privacy

The AI models (the answer model, the embedding model and image captioning) run entirely on your machine, and your conversation history stays in a local database. No queries go to an AI company. To answer a question, the app does send your search to DuckDuckGo and downloads the result pages, the same as searching in a browser would.

## Setup

### Requirements
- **Python 3.12**
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**, which runs the Postgres and Qdrant databases
- **About 15 GB of free disk space**, mostly for the model weights downloaded on first run
- **RAM:** 16 GB minimum, and 32 GB is much faster (see [Known limitations](#known-limitations))
- **Hardware:** an Apple Silicon Mac (uses the GPU through MPS), or a machine with an NVIDIA GPU (uses 4-bit quantization through CUDA). CPU-only machines work, but slowly.

### First-time setup
```bash
# 1. Get the code
git clone https://github.com/chiemc/ai_powered_search_engine.git
cd ai_powered_search_engine

# 2. Create your config file, then edit .env and choose a password
cp .env.example .env

# 3. Install Python dependencies into a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running it
```bash
# 1. Open Docker Desktop and wait until it's running, then start the databases
docker compose up -d

# 2. Activate the virtual environment (needed in every new terminal)
source .venv/bin/activate

# 3. Start the app, then open http://localhost:7860
python app.py
```
The first launch downloads about 8 GB of models, which can take a while. Later launches take about a minute.

**When you're done:** press `Ctrl+C` in the terminal running the app, then run `docker compose down` to stop the databases. Your data is kept for next time.

Adminer, a web interface for viewing the Postgres database, runs at http://localhost:8080 while the databases are up.

## Usage tips

The question is sent to the search engine as written, so phrase it the way you'd type into a search engine:
- **Name the subject every time.** Follow-up questions are not rewritten yet, so "When was it founded?" searches for exactly that. Write "When was Canberra founded?" instead.
- **Say whether you mean a character or the real person.** "John B and Sarah relationship Outer Banks" vs. "Chase Stokes girlfriend."
- **Ask one thing per question**, and include the year or season for anything that changes over time.
- Keep **Enable Rag** on. It gives the model the passages most relevant to your question.
- Answers cite sources inline ([1], [2]), and the links are listed under each answer. Click through to check important facts.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Connection refused` mentioning port 6333 or 5432 | Qdrant or Postgres isn't running | Start Docker Desktop, then run `docker compose up -d` |
| `address already in use` on port 7860 | An earlier copy of the app is still running | `lsof -ti :7860 \| xargs kill`, then start it again |
| `unknown shorthand flag: 'd' in -d` | Docker didn't recognize `compose`: a typo, or a terminal opened before Docker Desktop started | Retype the command by hand, or open a new terminal |
| `ModuleNotFoundError` | The virtual environment isn't active | `source .venv/bin/activate` |
| `password authentication failed` | `.env` was changed after the database was first created | `docker compose down -v` (this deletes stored history), then `docker compose up -d` |
| `403` / `402` / `405` / `Failed to resolve` warnings in the log | Some websites block bots, are paywalled, or no longer exist | Normal: those pages are skipped. If answers are thin, raise the results slider. |
| Answers take several minutes | Not enough RAM, so the system is swapping to disk | Close other apps (see [Known limitations](#known-limitations)) |


## How it was debugged

Getting this running took 17 fixes, including a native crash in the model loader, a search library that silently returned nothing, and a RAG pipeline that was giving the model the wrong text. That last one surfaced as confident but wrong answers, and I fixed it over three rounds of flagging bad answers and tracing their causes. The full write-up, with symptom, cause, fix and lesson for each problem, is in **[DEBUGGING.md](DEBUGGING.md)**.

## Known limitations

- **Slow on a 16 GB Mac: 4–5 minutes per answer.** Qwen at fp16 (about 6 GB), Florence-2 at fp32 on the CPU, LaBSE and Docker together pushed the system about 10 GB into swap. Web search and page fetching take only about 30 seconds. Nearly all the time goes to LLM generation, and that is slowed by swapping.
- **Follow-up questions are searched literally.** "When was it founded?" goes to the search engine word for word, so it returns unrelated results. The chat history reaches the LLM but not the search.
- **SQL is built with f-strings** in `memory.py`. Quotes are escaped, but parameterized queries would be safer.
- **Debug output:** `print` statements instead of a logging framework.

## Roadmap

1. **Speed:** serve the LLM through Ollama or MLX at 4-bit (about 2 GB instead of 6 GB, typically 3–5× faster on Apple Silicon); stream tokens to the UI; fetch pages in parallel; load Florence-2 only when an image is uploaded.
2. **Answer quality:** use the chat history to rewrite follow-up questions into standalone search queries; switch to a chat-style UI (`gr.ChatInterface`). (Numbered citations are done; see [DEBUGGING.md](DEBUGGING.md).)
3. **A browser assistant, in the spirit of Perplexity's Comet:** expose the pipeline as a FastAPI backend and build a Chrome sidebar extension that can summarize or answer questions about the current page.
