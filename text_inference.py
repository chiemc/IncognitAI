import warnings
warnings.filterwarnings("ignore")

import atexit
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from dotenv import load_dotenv
from memory import ConversationHistory, PGClient
import os
import random as r
from websearching import date_for_debug

load_dotenv()

# Qwen's official 4-bit GGUF builds, run with llama.cpp. llama.cpp uses Metal on Apple Silicon,
# CUDA on NVIDIA and the CPU elsewhere, with the same code. LLM_SIZE in .env picks the size;
# see .env.example for the tradeoffs.
MODELS = {
    "3b": {
        "label": "Qwen 2.5 3B",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
    },
    "7b": {
        "label": "Qwen 2.5 7B",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        # The 7B file is published in two parts; llama.cpp loads both from the first
        "filename": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "additional_files": ["qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"],
    },
}
llm_size = os.getenv("LLM_SIZE", "3b").strip().lower()
if llm_size not in MODELS:
    raise SystemExit(f"LLM_SIZE in .env must be one of {', '.join(MODELS)} (got '{llm_size}')")
model = MODELS[llm_size]
MODEL_LABEL = model["label"]

# hf_hub_download reuses files already on disk and works offline (HF_HUB_OFFLINE=1), unlike
# Llama.from_pretrained, which asks huggingface.co for the file list on every start
model_path = hf_hub_download(model["repo_id"], model["filename"])
for extra_file in model.get("additional_files", []):
    hf_hub_download(model["repo_id"], extra_file)  # lands next to the first part, where llama.cpp looks for it

llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,  # put every layer on the GPU when there is one
    n_ctx=8192,       # room for the search context plus recent history
    verbose=False,
)
# Free the model before Python shuts down: otherwise llama.cpp's Metal backend aborts on exit
# (GGML_ASSERT in ggml_metal_device_free) and prints a native backtrace every time the app stops
atexit.register(llm.close)

pg_db = os.getenv("PG_DB")
pg_user = os.getenv("PG_USER")
pg_psw = os.getenv("PG_PASSWORD")

pg_conn_str = f"postgresql+psycopg2://{pg_user}:{pg_psw}@localhost:5432/{pg_db}"
pg_client = PGClient(pg_conn_str)

usr_id = r.randint(1, 10000)
convo_hist = ConversationHistory(pg_client, usr_id)
SYSTEM_PROMPT = (
    "You are a web search assistant. Answer the user's QUESTION using only the CONTEXT in their latest message, "
    "which was taken from the web pages listed under SOURCES. "
    "Cite the sources you use inline with their numbers, like [1] or [2], but do not write a list of sources at the end. "
    "Answer only what was asked. Do not add facts that are not in the context, especially names of people, "
    "such as who plays a character. "
    "Keep the people in a story distinct from the actors who play them. "
    "Base each answer on the CONTEXT of the latest message, not on your earlier answers, which may have been wrong. "
    "Ignore any context that is not relevant to the question. "
    "The context may describe events that happened after your training data; trust it over your own knowledge. "
    "When the context clearly answers the question, give that answer directly and confidently. "
    "Only if the context does not answer the question, say so plainly instead of guessing. "
    "Never invent names, dates, relationships or other facts. Keep the answer concise."
)
convo_hist.add_message(role="system", content=SYSTEM_PROMPT)

# Only the most recent exchanges are sent back to the model, so old topics don't leak into new answers
MAX_HISTORY_MESSAGES = 4

def pipe(prompt: list, temperature: float, top_p: float, max_new_tokens: int, repetition_penalty: float):
    # llama.cpp applies Qwen's chat template from the GGUF file and returns only the new answer
    out = llm.create_chat_completion(
        messages=prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
    )
    return out["choices"][0]["message"]["content"]

def text_inference(message, debug, query=None):
    # History holds only past questions and answers; the search context is sent for the current turn only
    history = convo_hist.get_conversation_history()
    system, past = history[0], history[1:]
    prompt = [system] + past[-MAX_HISTORY_MESSAGES:] + [{"role": "user", "content": message}]
    if debug:
        print(f"[{date_for_debug()}] PROMPT")
        print(prompt)
    res = pipe(
        prompt,
        temperature=0.1,
        top_p=1,
        max_new_tokens=512,
        # A high penalty pushes the model to misspell names it has already seen in the context
        repetition_penalty=1.05
    )
    ret = res.strip()
    convo_hist.add_message(role="user", content=query or message)
    convo_hist.add_message(role="assistant", content=ret)
    return ret