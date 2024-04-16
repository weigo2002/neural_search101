from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = QdrantClient()

    def search(self, query: str):
        vector = self.model.encode(query).tolist()

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=5,
        )

        return [hit.payload for hit in search_result]
