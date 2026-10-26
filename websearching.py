import warnings
warnings.filterwarnings("ignore")

import nltk
from ddgs import DDGS
from rake_nltk import Rake
from boilerpy3 import extractors
import json
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from rag import NeuralSearcher
import random as rnd
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

# rake-nltk needs these NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

encoder = SentenceTransformer("sentence-transformers/LaBSE")
# Recursive splitting falls back to single newlines/spaces, so chunks actually stay near 1000 characters
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
extractor = extractors.ArticleExtractor()
# Identify the bot honestly (Wikipedia's robot policy rate-limits generic or spoofed user agents)
HEADERS = {"User-Agent": "PrAIvateSearch/0.1 (personal AI search project; python-requests)"}
collection_name = f"cute_kitty_{rnd.randint(1,10000)}"
qdrant_client = QdrantClient("http://localhost:6333")
searcher = NeuralSearcher(collection_name, qdrant_client, encoder)
r = Rake()

if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)

def extract_corpus(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.split('.')
    if len(domain) > 2:
        return domain[-2]
    return domain[0]

def upload_to_qdrant(client: QdrantClient, collection_name: str, encoder: SentenceTransformer, text: str):
    try:
        chunks = splitter.split_text(text)
        docs = []
        for chunk in chunks:
            docs.append({"text": chunk})
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

def web_search(query, num_results=5, enable_rag=False, debug = True):
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
    urls = list(set(search_results))
    jsonlike = {}
    finalcont = ""
    context = ""
    if len(urls) > 0:
        for url in urls:
            try:
                # boilerpy3's own downloader uses Python's default user agent, which many sites block (HTTP 403)
                resp = requests.get(url, headers=HEADERS, timeout=10)
                resp.raise_for_status()
                content = extractor.get_content(resp.text)
                r.extract_keywords_from_text(content)
                keywords = r.get_ranked_phrases()[:20]
                jsonlike.update({url: {"keywords": keywords}})
                finalcont += content + "\n\n"
            except Exception as e:
                if debug:
                    print(f"[{date_for_debug()}] WARNING! {e}")
                content = extract_corpus(url) + " " + " ".join(url.split("/")[3:])
                r.extract_keywords_from_text(content)
                keywords = r.get_ranked_phrases()[:20]
                jsonlike.update({url: {"keywords": keywords}})
                finalcont += content + "\n\n"
                continue
    else:
        jsonlike = {"keywords": "THE SEARCH DID NOT PRODUCE MEANINGFUL RESULTS (base the answer on the context, if given)"}
    if enable_rag and finalcont:
        res = searcher.search(finalcont, 3)
        for i in range(len(res)):
            context += res[i]["text"]+"\n\n"+"---------------"+"\n\n"
    truth = upload_to_qdrant(qdrant_client, collection_name, encoder, finalcont)
    jsonstr = json.dumps(jsonlike)
    if truth:
        if context:
            return "QUERY:\n\n"+query+"\n\nKEYWORDS:\n\n"+jsonstr+"\n\nCONTEXT:\n\n"+context, f"[{date_for_debug()}] SUCCESS! Semantic memory successfully updated!"
        else:
            return "QUERY:\n\n"+query+"\n\nKEYWORDS:\n\n"+jsonstr, f"[{date_for_debug()}] SUCCESS! Semantic memory successfuly updated!"
    if context:
        return "QUERY:\n\n"+query+"\n\nKEYWORDS:\n\n"+jsonstr+"\n\nCONTEXT:\n\n"+context, f"[{date_for_debug()}] WARNING! Something went wrong while updating semantic memory"
    return "QUERY:\n\n"+query+"\n\nKEYWORDS:\n\n"+jsonstr, f"[{date_for_debug()}] WARNING! Something went wrong while updating semantic memory"