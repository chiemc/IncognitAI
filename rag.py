from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

class NeuralSearcher:
    def __init__(self, collection_name: str, client: QdrantClient, model: SentenceTransformer):
        self.collection_name = collection_name
        self.qdrant_client = client
        self.model = model

    def search(self, text: str, limit: int = 3, urls: list = None):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Optionally only search chunks from the given pages
        query_filter = None
        if urls:
            query_filter = models.Filter(must=[models.FieldCondition(key="url", match=models.MatchAny(any=urls))])

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        ).points
        payloads = [hit.payload for hit in search_result]
        return payloads
