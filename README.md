# AI-Powered Private Search Engine

A self-hosted, Perplexity-style search engine. You ask a question, or upload an image. The app searches the web, reads the result pages, stores them in a vector database for retrieval-augmented generation (RAG), and writes an answer with a local LLM. All the models run on your own machine.

Based on the Hugging Face tutorial [*Building an AI-powered search engine from scratch*](https://huggingface.co/blog/as-cle-bert/build-an-ai-powered-search-engine-from-scratch) ([AstraBert/PrAIvateSearch](https://github.com/AstraBert/PrAIvateSearch)). The tutorial targets an older library stack and a CUDA GPU, so getting it running on an Apple Silicon Mac with current libraries took the debugging written up below.

## Architecture

```
            ┌──────────────┐
 query ───▶ │  Gradio UI   │  app.py
 (+ image)  └──────┬───────┘
                   │ image? ──▶ Florence-2 caption ──▶ RAKE keywords      image_gen.py
                   ▼
        ┌─────────────────────┐
        │  DuckDuckGo search  │ ──▶ fetch pages ──▶ boilerpy3 text extraction
        └─────────┬───────────┘                    + RAKE keyword extraction  websearching.py
                  │
                  ├──▶ chunk + embed (LaBSE) ──▶ Qdrant (semantic memory)
                  │                                  │
                  │      RAG: top-3 similar chunks ◀─┘                        rag.py
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
| Page text extraction | boilerpy3 + requests |
| Keyword extraction | rake-nltk |
| Embeddings | `sentence-transformers/LaBSE` |
| Vector DB | Qdrant (Docker) |
| Conversation memory | PostgreSQL 17 (Docker) + SQLAlchemy |
| LLM | `Qwen/Qwen2.5-3B-Instruct` (transformers; 4-bit on CUDA, fp16 on Apple MPS) |
| Image captioning | Florence-2-large (native transformers implementation) |

## Running it

Requirements: Python 3.12, Docker Desktop, about 10 GB of disk space for the models. You need 16 GB of RAM at minimum, and more is better (see [Known limitations](#known-limitations)).

```bash
# 1. Create a .env file with PG_USER, PG_PASSWORD and PG_DB

# 2. Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start Postgres, Qdrant and Adminer (Adminer is a DB admin UI on :8080)
docker compose up -d

# 4. Run the app, then open http://localhost:7860
python app.py
```

The first run downloads about 8 GB of model weights.

---

## Debugging log

These are the problems I hit going from "code written" to "working end to end," roughly in the order they surfaced. Each entry gives the symptom, the root cause, the fix, and what it taught me.

### 1. `SyntaxError: 'return' outside function` (rag.py)
- **Symptom:** `py_compile` failed on `rag.py`.
- **Cause:** I'd pasted the search logic straight into the `NeuralSearcher` class body. The class had no `__init__` and no `search` method, so the code that used `self` and `return` sat at class level.
- **Fix:** I added `__init__(collection_name, client, model)` and wrapped the logic in `search(text, limit)`.
- **Lesson:** Python runs a class body when the class is defined, and `self` only exists inside methods.

### 2. `NameError: context` on every successful search
- **Cause:** `context = ""` was only assigned in the `else` branch for "no search results." Any search that *did* return results later read an undefined variable.
- **Fix:** I initialize `context` at the top of the function.
- **Lesson:** A variable assigned on only some branches is a latent bug that only appears on the "happy path" you didn't test.

### 3. A typo'd environment variable produced a broken DB URL
- **Cause:** `os.getenv("PG_DUB")` instead of `PG_DB`. `getenv` returns `None` silently, so the connection string ended in `/None`.
- **Lesson:** For required configuration, `os.environ["KEY"]` fails loudly; `getenv` hides the mistake.

### 4. A malformed f-string in the debug timestamps
- **Cause:** `f"{date.minute:{date.second}}"`. The nested braces made `date.second` the *format width* for `minute`, rather than printing it after a colon.
- **Fix:** `f"{date.minute:02d}:{date.second:02d}"`.

### 5. Every Qdrant upload overwrote the previous one
- **Cause:** Point IDs came from `enumerate(docs)`, so every search wrote IDs 0..n and replaced the last search's chunks. RAG could only ever "remember" one search.
- **Fix:** Use `uuid.uuid4()` for point IDs.
- **Lesson:** In a key-value store, ID collisions don't raise errors. They overwrite data silently.

### 6. Foreign key order when resetting tables (memory.py)
- **Cause:** The code ran `DROP TABLE conversations` before `DROP TABLE messages`, but `messages` has a foreign key to `conversations`. The first drop failed, and the error was swallowed by a `try/except` that only warned, so stale tables survived.
- **Fix:** Drop the child table (`messages`) first.

### 7. Hardcoded CUDA on a Mac
- **Symptom:** `device_map="cuda:0"` plus bitsandbytes 4-bit quantization. My M1 Pro has no CUDA.
- **Fix:** Pick the device at runtime: CUDA with 4-bit quantization, then Apple MPS at fp16, then CPU at fp32. `bitsandbytes` is installed only on non-macOS systems, using a platform marker in `requirements.txt`.

### 8. Native crash: `malloc: double free` while loading the model
- **Symptom:** The process aborted partway through "Loading weights," with no Python traceback.
- **Investigation:** I wrote a minimal reproduction script that loaded only the model. `from_pretrained(..., device_map="mps")` crashed every time (exit code 139, a segfault). Loading on the CPU and then calling `.to("mps")` worked.
- **Cause:** A bug in transformers 5's device-map loading path on Apple MPS. It's in native code, which is why there was no Python stack trace.
- **Fix:** On MPS, load on the CPU first and then move the model to the GPU.
- **Lesson:** When a crash happens in native code, cut the program down to the smallest reproduction and change one variable at a time.

### 9. Breaking API changes in transformers 5
- `apply_chat_template(..., return_tensors="pt")` now returns a **dict** by default, so `model.generate(tokenized_chat)` broke. Fix: pass `return_dict=True` and call `generate(**inputs)`.
- **Wrong answer returned:** The code took `decoded_output.split("<|im_start|>assistant\n")[1]`. With conversation history in the prompt, that's the *first* assistant reply in the history, not the new one. Fix: decode only the newly generated tokens, `outputs[0][input_len:]`.
- `temperature` was being ignored because sampling wasn't enabled. Fix: `do_sample=True`.
- I removed TRL's `setup_chat_format`. It cleared Qwen's built-in chat template and resized the token embeddings, even though Qwen already uses the ChatML format it was adding.

### 10. Florence-2 remote code isn't compatible with transformers 5
- **Cause:** `microsoft/Florence-2-large` ships its own model code (`trust_remote_code=True`), written for older transformers versions. Transformers now includes Florence-2 natively.
- **Fix:** Use `Florence2ForConditionalGeneration` with the converted `florence-community/Florence-2-large` weights. This also removed the `einops` and `timm` dependencies.

### 11. SQLAlchemy 2.1 picked the wrong Postgres driver
- **Symptom:** `ModuleNotFoundError: No module named 'psycopg'`.
- **Cause:** SQLAlchemy 2.1 maps `postgresql://` to psycopg **3** by default, but the project installs `psycopg2-binary`.
- **Fix:** Name the driver in the URL: `postgresql+psycopg2://...`.

### 12. Google search silently returned zero results
- **Symptom:** Every answer said "the search did not produce meaningful results."
- **Investigation:** I called `googlesearch.search()` directly and got `[]`. A raw request to Google returned a page with no result markup.
- **Cause:** Google now blocks scraping libraries like `googlesearch-python`, and the library fails silently instead of raising an error.
- **Fix:** Switch to `ddgs` (DuckDuckGo, no API key). It occasionally returns "No results found" for valid queries, so I added up to 3 retries.
- **Lesson:** Scraping-based dependencies can break without any code change on your side. Fail loudly, and test each external dependency on its own.

### 13. HTTP 403 when fetching result pages
- **Cause:** boilerpy3's `get_content_from_url` uses Python's default `urllib` user agent, which sites like Wikipedia block. Later, a spoofed browser user agent got rate-limited too, with a message pointing to Wikipedia's robot policy.
- **Fix:** Download pages with `requests` using an honest, descriptive user agent (`PrAIvateSearch/0.1 (...)`) and a timeout, then pass the HTML to `extractor.get_content()`. If a page still fails, the code falls back to keywords taken from the URL.
- **Lesson:** Read the robot policy. Identifying your client honestly works better than impersonating a browser.

### 14. RAG chunks were 27× too big
- **Symptom:** `Created a chunk of size 26962, which is longer than the specified 1000`.
- **Cause:** `CharacterTextSplitter` only splits on its separator (`"\n\n"`), and the extracted text didn't contain any. LaBSE only reads about the first 512 tokens of each input, so most of each chunk was never embedded at all.
- **Fix:** `RecursiveCharacterTextSplitter`, which falls back to `"\n"`, then spaces, then characters.
- **Result:** Follow-up queries now pull in relevant context from earlier searches.

### 15. Infrastructure issues
- **compose.yaml:** a `networds:` typo; a `ui` service that built from a `chatbot-ui/` directory that didn't exist; and an unpinned `postgres` image. Postgres 18+ won't start with a volume mounted at `/var/lib/postgresql/data`. Fix: remove the `ui` service and pin `postgres:17`.
- **`unknown shorthand flag: 'd' in -d`:** Docker shows this when it doesn't recognize the subcommand, e.g. a typo in `compose` or a terminal that can't see the Compose plugin. The message is misleading because it points at `-d`, not the actual problem.
- **Docker Desktop crashed** during testing because of memory pressure (see below).

### 16. Missing dependencies
The original `requirements.txt` was missing `gradio`, `pillow`, `accelerate`, `nltk`, `rake-nltk` and `boilerpy3`. rake-nltk also needs NLTK data (`stopwords`, `punkt_tab`) at runtime, which the code now downloads at startup.

---

## Known limitations

- **Slow on a 16 GB Mac: 4–5 minutes per answer.** Qwen at fp16 (about 6 GB), Florence-2 at fp32 on the CPU, LaBSE and Docker together pushed the system about 10 GB into swap. Web search and page fetching take only about 30 seconds. Nearly all the time goes to LLM generation, and that is slowed by swapping.
- **Follow-up questions are searched literally.** "When was it founded?" goes to the search engine word for word, so it returns unrelated results. The chat history reaches the LLM but not the search.
- **SQL is built with f-strings** in `memory.py`. Quotes are escaped, but parameterized queries would be safer.
- **Debug output:** `print` statements instead of a logging framework.

## Roadmap

1. **Speed:** serve the LLM through Ollama or MLX at 4-bit (about 2 GB instead of 6 GB, typically 3–5× faster on Apple Silicon); stream tokens to the UI; fetch pages in parallel; load Florence-2 only when an image is uploaded.
2. **Answer quality:** numbered citations ([1], [2]); use the chat history to rewrite follow-up questions into standalone search queries; switch to a chat-style UI (`gr.ChatInterface`).
3. **A browser assistant, in the spirit of Perplexity's Comet:** expose the pipeline as a FastAPI backend and build a Chrome sidebar extension that can summarize or answer questions about the current page.
