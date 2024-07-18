import time
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client import models
from data_reader import DataReader


def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


batch_size = 1000
collection_name = "binary-quantization-collection"
db_file = "./data/pubmed_abstracts_2024.db"
h5_file = "./data/pubmed_embeddings.h5"


# collect to our Qdrant Server
client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=True,
    timeout=1000,
)

reader = DataReader(h5_file, db_file)

doc_vectors = reader.doc_vectors
doc_ids = reader.doc_ids
num_rows = reader.num_rows

print('XXX_total_vectors: ', num_rows)

start_time = time.time()

result = client.search(
    collection_name=collection_name,
    query_vector=doc_vectors[0],
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=False,
        )
    ),
    timeout=1000,
)

end_time = time.time()
elapsed_time = end_time - start_time

print(result)

print(f"XXX_Elapsed time: {elapsed_time:.10f} seconds")

