# IncognitAI

**An AI-powered private search engine.**

A self-hosted, Perplexity-style search engine. You ask a question, or upload an image. The app searches the web, reads the result pages, stores them in a vector database for retrieval-augmented generation (RAG), and writes an answer with a local LLM. All the models run on your own machine.

Based on the Hugging Face tutorial [*Building an AI-powered search engine from scratch*](https://huggingface.co/blog/as-cle-bert/build-an-ai-powered-search-engine-from-scratch) ([AstraBert/PrAIvateSearch](https://github.com/AstraBert/PrAIvateSearch)). The tutorial targets an older library stack and a CUDA GPU, so getting it running on an Apple Silicon Mac with current libraries took the debugging written up in [DEBUGGING.md](DEBUGGING.md).

## Architecture

```
            ┌──────────────┐
 query ───▶ │  Gradio UI   │  app.py
 (+ image)  └──────┬───────┘
                   │ image? ──▶ Florence-2 caption, added to the question  image_gen.py
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
| Embeddings | `sentence-transformers/LaBSE` |
| Vector DB | Qdrant (Docker) |
| Conversation memory | PostgreSQL 17 (Docker) + SQLAlchemy |
| LLM | Qwen2.5-Instruct 3B (default) or 7B, 4-bit GGUF, run with `llama-cpp-python` |
| Image captioning | Florence-2-large (native transformers implementation) |

## Privacy

The AI models (the answer model, the embedding model and image captioning) run entirely on your machine, and your conversation history stays in a local database. No queries go to an AI company. To answer a question, the app does send your search to DuckDuckGo and downloads the result pages, the same as searching in a browser would.

After the first run, the models load from your disk without contacting Hugging Face. You can check this by starting the app with downloads blocked: `HF_HUB_OFFLINE=1 python app.py`.

## Setup

### Requirements
- **Python 3.12**
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**, which runs the Postgres and Qdrant databases
- **About 10 GB of free disk space**, mostly for the model weights downloaded on first run
- **RAM:** 16 GB recommended, or 32 GB to use the larger 7B answer model
- **Hardware:** an Apple Silicon Mac (the answer model uses the GPU through Metal automatically), or a machine with an NVIDIA GPU (see the note under setup). CPU-only machines work, but more slowly.
- **A C/C++ compiler,** because `pip` may build `llama-cpp-python` from source. Macs have one once Xcode Command Line Tools are installed (`xcode-select --install`). Windows needs the Visual Studio Build Tools.

### First-time setup
```bash
# 1. Get the code
git clone https://github.com/chiemc/IncognitAI.git
cd IncognitAI

# 2. Create your config file, then edit .env: choose a password and, optionally, the model size
cp .env.example .env

# 3. Install Python dependencies into a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Installing `llama-cpp-python` can take a few minutes, because it compiles llama.cpp for your machine. On a Mac it is built with Metal (GPU) support automatically. **NVIDIA users:** to run the answer model on the GPU, install it with CUDA support before the other requirements:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### Choosing the answer model
`LLM_SIZE` in `.env` selects which Qwen model writes the answers. Restart the app after changing it.

| | `3b` (default) | `7b` |
|---|---|---|
| Memory | ~2.6 GB | ~5.5 GB (32 GB of RAM recommended) |
| Answer generation | ~2–6 s | roughly twice as long |
| Download | 2.1 GB | 4.7 GB more |
| Accuracy | Good, but sometimes misreads relationships in its sources | Better reading comprehension; still makes occasional mistakes |

The app shows which model is active under its title.

**Going back to 3B and reclaiming the disk space:**
1. Set `LLM_SIZE=3b` in `.env` and restart the app. Do this first: otherwise the next start downloads the 7B again.
2. Delete the 7B model files (4.7 GB), with the virtual environment active:
   ```bash
   hf cache ls                                                   # list stored models and their sizes
   hf cache rm model/Qwen/Qwen2.5-7B-Instruct-GGUF --dry-run     # preview; deletes nothing
   hf cache rm model/Qwen/Qwen2.5-7B-Instruct-GGUF               # delete (asks to confirm)
   ```
3. Run `hf cache ls` again. The 7B should be gone, and `Qwen2.5-3B-Instruct-GGUF` (the default model) should still be listed.

If you deleted the wrong model by mistake, nothing is lost for good: the app downloads whichever model it needs on its next start.

### Running it
```bash
# 1. Open Docker Desktop and wait until it's running, then start the databases
docker compose up -d

# 2. Activate the virtual environment (needed in every new terminal)
source .venv/bin/activate

# 3. Start the app, then open http://localhost:7860
python app.py
```
The first launch downloads about 5.5 GB of models (Qwen 2.1 GB, LaBSE and Florence-2), which can take a while. Later launches take about 15–30 seconds.

**When you're done:** press `Ctrl+C` in the terminal running the app, then run `docker compose down` to stop the databases. Your data is kept for next time.

Adminer, a web interface for viewing the Postgres database, runs at http://localhost:8080 while the databases are up.

## Usage tips

The question is sent to the search engine as written, so phrase it the way you'd type into a search engine:
- **Name the subject every time.** Follow-up questions are not rewritten yet, so "When was it founded?" searches for exactly that. Write "When was Canberra founded?" instead.
- **Say whether you mean a character or the real person.** "John B and Sarah relationship Outer Banks" vs. "Chase Stokes girlfriend."
- **Ask one thing per question**, and include the year or season for anything that changes over time.
- **For image searches, also type a question** (e.g. "What is this company's internship program about?"). The question and the image description are searched together.
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
| Answers take minutes instead of seconds | Not enough free RAM, so the system is swapping to disk | Close other apps, and check that only one copy of the app is running: `lsof -i :7860` |


## How it was debugged

Getting this running took 17 fixes, including a native crash in the model loader, a search library that silently returned nothing, and a RAG pipeline that was giving the model the wrong text. That last one surfaced as confident but wrong answers, and I fixed it over three rounds of flagging bad answers and tracing their causes. The full write-up, with symptom, cause, fix and lesson for each problem, is in **[DEBUGGING.md](DEBUGGING.md)**.

## Known limitations

- **About 10–20 seconds per text answer on a 16 GB M1 Pro, plus about 9 seconds when an image is attached.** That compares with a few seconds for hosted AI search. Measured breakdown: web search ~2.5 s, downloading and extracting 5 pages ~4 s, embedding ~2 s, generating the answer ~2–6 s, captioning an image ~8.5 s. Most of the time now goes to the web, not the model. If memory runs short (for example, an accidental second copy of the app), the system swaps to disk and answers can take minutes.
- **Image search describes images; it can't recognize faces.** Florence-2 can read text in an image and describe what's in it ("a man with a beard"), but it can't tell you *who* someone is. A screenshot works when the person's name is visible on screen.
- **Follow-up questions are searched literally.** "When was it founded?" goes to the search engine word for word, so it returns unrelated results. The chat history reaches the LLM but not the search.
- **SQL is built with f-strings** in `memory.py`. Quotes are escaped, but parameterized queries would be safer.
- **Debug output:** `print` statements instead of a logging framework.

## Roadmap

1. **Speed:** ~~run Qwen at 4-bit~~ (done: switched to `llama-cpp-python`, see [DEBUGGING.md](DEBUGGING.md)); download pages in parallel (the largest remaining cost); stream tokens to the UI so text appears as it's written; run Florence-2 on the GPU, or load it only when an image is uploaded.
2. **Answer quality:** use the chat history to rewrite follow-up questions into standalone search queries; switch to a chat-style UI (`gr.ChatInterface`). (Numbered citations are done; see [DEBUGGING.md](DEBUGGING.md).)
3. **A browser assistant, in the spirit of Perplexity's Comet:** expose the pipeline as a FastAPI backend and build a Chrome sidebar extension that can summarize or answer questions about the current page.
