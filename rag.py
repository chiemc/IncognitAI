from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class NeuralSearcher:
    def __init__(self, collection_name: str, client: QdrantClient, model: SentenceTransformer):
        self.collection_name = collection_name
        self.qdrant_client = client
        self.model = model

    def search(self, text: str, limit: int = 3):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None, # If you don't want any filters for now
            limit=limit,
        ).points
        payloads = [hit.payload for hit in search_result]
        return payloads
