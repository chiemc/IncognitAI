import warnings
warnings.filterwarnings("ignore")

from ddgs import DDGS
import trafilatura
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from rag import NeuralSearcher
import random as rnd
import re
import time
import uuid
from datetime import datetime

# How many of the most relevant passages (about 1000 characters each) the model reads per question
RAG_PASSAGES = 5

encoder = SentenceTransformer("sentence-transformers/LaBSE")
# Recursive splitting falls back to single newlines/spaces, so chunks actually stay near 1000 characters
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Identify the bot honestly (Wikipedia's robot policy rate-limits generic or spoofed user agents)
HEADERS = {"User-Agent": "IncognitAI/0.1 (personal AI search project; python-requests)"}
collection_name = f"cute_kitty_{rnd.randint(1,10000)}"
qdrant_client = QdrantClient("http://localhost:6333")
searcher = NeuralSearcher(collection_name, qdrant_client, encoder)

if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)

# Photo credits like "ROBERT VIGLASKY/NETFLIX" sit next to captions about the people pictured,
# and the model misread the photographer as the actor; drop "First Last/Outlet" credits for common outlets
PHOTO_OUTLETS = (
    "Netflix|Getty Images|WireImage|FilmMagic|Shutterstock|Alamy|Everett Collection|Kobal|AP|AFP|Reuters|EPA|"
    "HBO|Max|BBC|Disney\\+?|Hulu|Apple TV\\+?|Amazon|Prime Video|Paramount\\+?|Peacock|NBC|CBS|ABC|FOX|"
    "Warner Bros\\.?|Universal|Sony|Lionsgate|A24|Instagram|Bloomberg"
)
PHOTO_CREDIT = re.compile(rf"\b(?:[A-Z][\w.'’-]+ ){{1,3}}[A-Z][\w.'’-]+/(?:{PHOTO_OUTLETS})(?=\b|[A-Z])", re.IGNORECASE)

def strip_photo_credits(text: str) -> str:
    return PHOTO_CREDIT.sub("", text)

def upload_to_qdrant(client: QdrantClient, collection_name: str, encoder: SentenceTransformer, pages: list):
    try:
        docs = []
        for url, text in pages:
            for chunk in splitter.split_text(text):
                # Keep the source URL with each chunk so answers can cite it
                docs.append({"text": chunk, "url": url})
        if not docs:
            return False
        client.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=encoder.encode(doc["text"]).tolist(),
                    payload=doc,
                )
                for doc in docs
            ],
        )
        return True
    except Exception as e:
        return False

def date_for_debug():
    date = datetime.now()
    s = f"{date.year}-{date.month}-{date.day} {date.hour}:{date.minute:02d}:{date.second:02d}"
    return s

def web_search(query, num_results=5, debug = True):
    global qdrant_client, encoder, collection_name
    search_results = []
    # Google now blocks scraping libraries like googlesearch-python, so use DuckDuckGo instead
    # DuckDuckGo occasionally returns "No results found" for a valid query, so retry a couple of times
    for attempt in range(3):
        try:
            for result in DDGS().text(query, max_results=num_results):
                search_results.append(result["href"])
            break
        except Exception as e:
            if debug:
                print(f"[{date_for_debug()}] WARNING! Web search failed (attempt {attempt + 1}/3): {e}")
            time.sleep(1)
    urls = list(dict.fromkeys(search_results))
    pages = []
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            # trafilatura extracts the main article text; boilerpy3 sometimes returned embedded JSON/script data instead
            content = strip_photo_credits(trafilatura.extract(resp.text) or "").strip()
            if content:
                pages.append((url, content))
        except Exception as e:
            if debug:
                print(f"[{date_for_debug()}] WARNING! {e}")

    # Store this search's pages BEFORE retrieving, so RAG can find them (not just chunks from older searches)
    truth = upload_to_qdrant(qdrant_client, collection_name, encoder, pages)

    # Number each source so the model can cite it as [1], [2], ...
    sources = []
    def source_num(url):
        if url not in sources:
            sources.append(url)
        return sources.index(url) + 1

    context = ""
    if pages:
        # RAG: retrieve the chunks most similar to the user's question, from this search's pages only
        # (chunks left over from earlier searches kept leaking unrelated topics into answers)
        for hit in searcher.search(query, RAG_PASSAGES, urls=[url for url, _ in pages]):
            context += f"[{source_num(hit['url'])}] {hit['text']}\n\n"

    # The model doesn't know the current date, so it treats recent years as "the future" without this
    today = f"TODAY'S DATE: {datetime.now():%B %d, %Y}\n\n"
    if context:
        source_list = "\n".join(f"[{i + 1}] {url}" for i, url in enumerate(sources))
        prompt = f"{today}QUESTION:\n{query}\n\nSOURCES:\n{source_list}\n\nCONTEXT:\n{context}"
    else:
        prompt = f"{today}QUESTION:\n{query}\n\nCONTEXT:\nNo web pages could be retrieved for this question."

    if truth:
        return prompt, f"[{date_for_debug()}] SUCCESS! Semantic memory successfully updated!", sources
    return prompt, f"[{date_for_debug()}] WARNING! Something went wrong while updating semantic memory", sources