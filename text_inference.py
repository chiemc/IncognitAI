import warnings
warnings.filterwarnings("ignore")

import accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
from memory import ConversationHistory, PGClient
import os
import random as r
from websearching import date_for_debug

load_dotenv()

model_name = "Qwen/Qwen2.5-3B-Instruct"

# bitsandbytes 4-bit quantization only works on CUDA; fall back to MPS (Apple Silicon) or CPU unquantized
if torch.cuda.is_available():
    device = "cuda:0"
    dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type= "nf4"
    )
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    quantization_config = None
else:
    device = "cpu"
    dtype = torch.float32
    quantization_config = None

if quantization_config is not None:
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, dtype=dtype, quantization_config=quantization_config)
else:
    # device_map="mps" crashes the loader (double free) on Apple Silicon, so load on CPU and move it afterwards
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
# Qwen2.5 already ships a ChatML chat template, so no extra chat formatting is needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

pg_db = os.getenv("PG_DB")
pg_user = os.getenv("PG_USER")
pg_psw = os.getenv("PG_PASSWORD")

pg_conn_str = f"postgresql+psycopg2://{pg_user}:{pg_psw}@localhost:5432/{pg_db}"
pg_client = PGClient(pg_conn_str)

usr_id = r.randint(1, 10000)
convo_hist = ConversationHistory(pg_client, usr_id)
convo_hist.add_message(role="system", content="You are a websearching assistant: your task is to create a human-readable content based on a JSON representation of the keywords of several websites related to the search that the user performed and on the context that you are provided with")

def pipe(prompt: str, temperature: float, top_p: float, max_new_tokens: int, repetition_penalty: float):
    inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(quantized_model.device)
    outputs = quantized_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    # Decode only the newly generated tokens, not the prompt/history
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    results = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return results

def text_inference(message, debug):
    convo_hist.add_message(role="user", content=message)
    prompt = convo_hist.get_conversation_history()
    if debug:
        print(f"[{date_for_debug()}] CONVERSATIONAL HISTORY")
        print(prompt)
    res = pipe(
        prompt,
        temperature=0.1,
        top_p=1,
        max_new_tokens=512,
        repetition_penalty=1.2
    )
    ret = res.strip()
    convo_hist.add_message(role="assistant", content=ret)
    return ret