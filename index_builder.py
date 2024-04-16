from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm
from qdrant_client.models import VectorParams, Distance

class IndexBuilder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

    def index_vectors(self, collection_name):
        df = pd.read_json("./startups_demo.json", lines=True)
        vectors = self.model.encode(
            [row.alt + ". " + row.description for row in df.itertuples()],
            show_progress_bar=True,
        )
        client = QdrantClient("http://localhost:6333")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        fd = open("./startups_demo.json")
        payload = map(json.loads, fd)

        client.upload_collection(
            collection_name=collection_name,
            vectors=vectors,
            payload=payload,
            ids=None,
            batch_size=256,
        )