import time
import logging
from pprint import pprint
from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:

    def __init__(self, embed_model_name="nomic-ai/nomic-embed-text-v1.5", collection_name="binary-quantization-collection"):
        self.embed_model_name = embed_model_name
        self.collection_name = collection_name
        self.embed_model = self._load_embed_model()
        self.qdrant_client = self._set_qdrant_client()

    def _set_qdrant_client(self):
        client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=True,
            timeout=1000,
        )
        return client

    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True, cache_folder='./hf_cache')
        return embed_model

    def search(self, query, top_k=20):
        query_embedding = self.embed_model.get_query_embedding(query)
        
        # Start the timer
        start_time = time.time()

        result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=False,  # Set to False if you don't need rescoring to improve speed
                    oversampling=1.5,  # Adjust this value for better performance
                )
            ),
            limit=top_k,  # Ensure the limit is set to the number of results you need
            timeout=500,  # Reduce timeout if possible
        )
        
        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logger.info(f"Execution time for the search: {elapsed_time:.4f} seconds")

        return result
    
