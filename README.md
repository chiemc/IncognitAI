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
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**, which runs the two databases (Postgres and Qdrant)
- **About 10 GB of free disk space** for the AI models, Python packages and databases
- **16 GB of RAM** (32 GB if you want the larger 7B answer model; see [below](#optional-choosing-the-answer-model))
- **A C/C++ compiler.** One package, `llama-cpp-python`, is compiled on your computer during installation.
  - **Mac:** run `xcode-select --install` once, if you haven't already.
  - **Windows:** install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" option.
  - **Linux:** usually already installed (`gcc`).

The app uses your GPU automatically on Apple Silicon Macs. On a computer with an NVIDIA GPU, it uses the GPU if you follow the NVIDIA step below. Other computers run on the CPU, which works but is slower.

### Install (once)
The commands below are for macOS and Linux. On Windows, use `copy` instead of `cp` and `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

**1. Download the code**
```bash
git clone https://github.com/chiemc/IncognitAI.git
cd IncognitAI
```

**2. Create your settings file**, then open `.env` and change `PG_PASSWORD` to a password of your choice
```bash
cp .env.example .env
```

**3. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Your prompt now starts with `(.venv)`. Run `python3 --version` first if you're unsure which Python you have: it needs to be 3.12.

**4. Install the Python packages**

*Mac, or a computer without an NVIDIA GPU:*
```bash
pip install -r requirements.txt
```

*Computer with an NVIDIA GPU:* run these two commands in this order. The first builds `llama-cpp-python` with GPU support; the second installs everything else.
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python==0.3.36
pip install -r requirements.txt
```

Expect this step to take a few minutes. Most of that is compiling `llama-cpp-python`, and it may look stuck at "Building wheel for llama-cpp-python" for a minute or two. That's normal.

### Run it
Each time you want to use the app:
```bash
# 1. Open Docker Desktop and wait until it's running, then start the databases
docker compose up -d

# 2. Activate the virtual environment (needed in every new terminal)
source .venv/bin/activate

# 3. Start the app
python app.py
```
Then open **http://localhost:7860** in your browser.

The first start downloads about 5.5 GB of AI models, so it can take a while. Later starts take about 15–30 seconds.

**When you're done:** press `Ctrl+C` in the terminal running the app, then run `docker compose down` to stop the databases. Your data is kept for next time.

Adminer, a web page for looking inside the Postgres database, runs at http://localhost:8080 while the databases are up.

### Optional: choosing the answer model
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
| `Failed building wheel for llama-cpp-python` during install | No C/C++ compiler | Install one (see [Requirements](#requirements)), then run the install step again |
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
