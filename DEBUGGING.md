# Debugging Log

These are the problems I hit getting this project from "code written" to "working end to end, with accurate answers." It started from the [Hugging Face tutorial](https://huggingface.co/blog/as-cle-bert/build-an-ai-powered-search-engine-from-scratch), which targets an older library stack and an NVIDIA GPU. I ran it on an Apple Silicon Mac (M1 Pro, 16 GB) with current versions of transformers, SQLAlchemy and Qdrant.

Each entry gives the symptom, the root cause, the fix, and what it taught me. The biggest problems come first.

**Contents**
- [Highlights](#highlights)
  - [The app ran, but its answers were confidently wrong](#the-app-ran-but-its-answers-were-confidently-wrong)
  - [Native crash: `malloc: double free` while loading the model](#native-crash-malloc-double-free-while-loading-the-model)
  - [Google search silently returned zero results](#google-search-silently-returned-zero-results)
  - [RAG chunks were 27× too big](#rag-chunks-were-27-too-big)
- [Other fixes](#other-fixes)
- [Smaller fixes](#smaller-fixes)

---

## Highlights

### The app ran, but its answers were confidently wrong

**Symptom:** I used Gradio's "Flag" button to save bad answers to a dataset, then reviewed them. They had invented plot points, wrong actor-to-role mappings, misspelled names ("Strokes" for Stokes), and one answer drifted into an earlier search's topic.

**Round 1: what text was the model actually given?**
1. The LLM was only shown RAKE keyword fragments from each page, never the page text, and the system prompt told it to "create content" from them. So it filled the gaps by inventing things.
2. RAG retrieved *before* storing the current pages, so it could only return chunks from earlier searches, and it used the whole page dump as the search query instead of the user's question. That's how an earlier search's topic leaked into an unrelated answer.
3. Every full past prompt (keywords and context) stayed in the chat history, so old topics kept reappearing and the prompt grew with every turn.
4. `repetition_penalty=1.2` discourages repeating tokens the model has already seen, including names in the context, so it misspelled them.

**Fixes:** Store pages first, then retrieve the top-5 chunks using the user's question. Give the model actual page text with numbered sources, and require inline citations ([1], [2]). Rewrite the system prompt to answer only from the context and say when it doesn't know. Store only the questions and answers in history, and send just the last 2 exchanges. Lower the repetition penalty to 1.05.

**Round 2: the right source was there but unreadable.** A new flagged answer confused a character's partner with the actor's real-life partner. The log showed that the correct source's text was mostly raw page code (`<p>`, `srcset`, `"button_url"`). boilerpy3 had extracted a block of embedded JSON from a `<script>` tag instead of the article: 127,000 characters of junk. Switching to `trafilatura` gave about 9,000 characters of clean article text from the same page. The app also now appends the real source links under each answer, instead of relying on the model to list them.

**Round 3: overcorrection.** The stricter prompt made the model too cautious. Asked "Was Drake at Oktoberfest in 2025?", it answered "No… he attended Oktoberfest in 2025… but the question asks about 2025," even though the sources clearly said yes. The model was trained before 2025 and has no sense of today's date, so it treated 2025 as the future. Fix: put today's date in every prompt, and tell the model to trust the context over its own knowledge.

One attempted fix made things worse. I added "start yes/no answers with Yes or No," and the model then began "Who directed Oppenheimer?" with "No" and argued itself out of the right answer. I removed the instruction and re-ran a fixed set of known-answer questions (a simple fact, character vs. actor, a detail buried mid-article, and a question with no answer in the sources). All four passed.

**Lessons:**
- "The model is too small" is often not the real reason a RAG system answers badly. Check what text actually reaches the model before blaming the model.
- Small models follow instructions very literally. Treat prompt changes like code changes and re-test against known-answer questions every time.

### Native crash: `malloc: double free` while loading the model

**Symptom:** The process aborted partway through "Loading weights," with no Python traceback.

**Investigation:** I wrote a minimal reproduction script that loaded only the model and tried two loading strategies. `from_pretrained(..., device_map="mps")` crashed every time (exit code 139, a segfault). Loading on the CPU and then calling `.to("mps")` worked.

**Cause:** A bug in transformers 5's device-map loading path on Apple MPS. It's in native code, which is why there was no Python stack trace.

**Fix:** On MPS, load on the CPU first and then move the model to the GPU.

**Lesson:** When a crash happens in native code, cut the program down to the smallest reproduction and change one variable at a time.

### Google search silently returned zero results

**Symptom:** Every answer said "the search did not produce meaningful results."

**Investigation:** I called `googlesearch.search()` directly and got `[]`. A raw request to Google returned a page with no result markup.

**Cause:** Google now blocks scraping libraries like `googlesearch-python`, and the library fails silently instead of raising an error.

**Fix:** Switch to `ddgs` (DuckDuckGo, no API key). It occasionally returns "No results found" for a valid query (the identical query worked a second later), so I added up to 3 retries.

**Lesson:** Scraping-based dependencies can break without any code change on your side. Fail loudly, and test each external dependency on its own.

### RAG chunks were 27× too big

**Symptom:** `Created a chunk of size 26962, which is longer than the specified 1000`.

**Cause:** `CharacterTextSplitter` only splits on its separator (`"\n\n"`), and the extracted text didn't contain any. LaBSE only reads about the first 512 tokens of each input, so most of each chunk was never embedded at all.

**Fix:** `RecursiveCharacterTextSplitter`, which falls back to `"\n"`, then spaces, then characters.

**Result:** Retrieval started returning passages that were actually relevant to the question.

---

## Other fixes

### Every Qdrant upload overwrote the previous one
Point IDs came from `enumerate(docs)`, so every search wrote IDs 0..n and replaced the last search's chunks. RAG could only ever "remember" one search. **Fix:** `uuid.uuid4()` for point IDs. **Lesson:** In a key-value store, ID collisions don't raise errors. They overwrite data silently.

### Breaking API changes in transformers 5
- `apply_chat_template(..., return_tensors="pt")` now returns a **dict** by default, so `model.generate(tokenized_chat)` broke. **Fix:** pass `return_dict=True` and call `generate(**inputs)`.
- **The wrong answer was returned:** the code took `decoded_output.split("<|im_start|>assistant\n")[1]`. With conversation history in the prompt, that's the *first* assistant reply in the history, not the new one. **Fix:** decode only the newly generated tokens, `outputs[0][input_len:]`.
- `temperature` was being ignored because sampling wasn't enabled. **Fix:** `do_sample=True`.
- I removed TRL's `setup_chat_format`. It cleared Qwen's built-in chat template and resized the token embeddings, even though Qwen already uses the ChatML format it was adding.

### Hardcoded CUDA on a Mac
The code used `device_map="cuda:0"` with bitsandbytes 4-bit quantization, and my M1 Pro has no CUDA. **Fix:** pick the device at runtime: CUDA with 4-bit quantization, then Apple MPS at fp16, then CPU at fp32. `bitsandbytes` is installed only on non-macOS systems, using a platform marker in `requirements.txt`.

### Florence-2 remote code isn't compatible with transformers 5
`microsoft/Florence-2-large` ships its own model code (`trust_remote_code=True`), written for older transformers versions, and transformers now includes Florence-2 natively. **Fix:** `Florence2ForConditionalGeneration` with the converted `florence-community/Florence-2-large` weights. This also removed the `einops` and `timm` dependencies.

### SQLAlchemy 2.1 picked the wrong Postgres driver
`ModuleNotFoundError: No module named 'psycopg'`. SQLAlchemy 2.1 maps `postgresql://` to psycopg **3** by default, but the project installs `psycopg2-binary`. **Fix:** name the driver in the URL: `postgresql+psycopg2://...`.

### HTTP 403 when fetching result pages
boilerpy3's `get_content_from_url` uses Python's default `urllib` user agent, which sites like Wikipedia block. A spoofed browser user agent got rate-limited too, with a message pointing to Wikipedia's robot policy. **Fix:** download pages with `requests` using an honest, descriptive user agent (`PrAIvateSearch/0.1 (...)`) and a timeout. Pages that still fail (bot protection, paywalls, dead links) are skipped. **Lesson:** Read the robot policy. Identifying your client honestly works better than impersonating a browser.

### Foreign key order when resetting tables
The code ran `DROP TABLE conversations` before `DROP TABLE messages`, but `messages` has a foreign key to `conversations`. The first drop failed, and the error was swallowed by a `try/except` that only warned, so stale tables survived. **Fix:** drop the child table first.

### Infrastructure
- **compose.yaml:** a `networds:` typo; a `ui` service that built from a `chatbot-ui/` directory that didn't exist; and an unpinned `postgres` image. Postgres 18+ won't start with a volume mounted at `/var/lib/postgresql/data`. **Fix:** remove the `ui` service and pin `postgres:17`.
- **`unknown shorthand flag: 'd' in -d`:** Docker shows this when it doesn't recognize the subcommand, e.g. a typo in `compose` or a terminal that can't see the Compose plugin. The message is misleading because it points at `-d`, not the actual problem.
- **Docker Desktop crashed** during testing because of memory pressure. Qwen, Florence-2, LaBSE and Docker together pushed a 16 GB machine about 10 GB into swap.

---

## Smaller fixes

- **`SyntaxError: 'return' outside function` in rag.py:** the search logic had been pasted directly into the class body, with no `__init__` or `search` method. Python runs a class body when the class is defined, and `self` only exists inside methods.
- **`NameError: context` on every successful search:** `context` was only assigned in the "no results" branch. A variable assigned on only some branches is a latent bug.
- **A typo'd environment variable:** `os.getenv("PG_DUB")` silently returned `None`, so the database URL ended in `/None`. For required configuration, `os.environ["KEY"]` fails loudly.
- **A malformed f-string:** in `f"{date.minute:{date.second}}"`, the nested braces made `second` the *format width* for `minute`.
- **Missing dependencies:** `gradio`, `pillow`, `accelerate`, `nltk`, `rake-nltk` and a page-text extractor were missing from `requirements.txt`. rake-nltk also needs NLTK data (`stopwords`, `punkt_tab`), which the code now downloads at startup.
