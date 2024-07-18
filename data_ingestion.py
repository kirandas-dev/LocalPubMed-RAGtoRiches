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
)

reader = DataReader(h5_file, db_file)

doc_vectors = reader.doc_vectors
doc_ids = reader.doc_ids
num_rows = reader.num_rows

if not client.collection_exists(collection_name=collection_name):

    client.create_collection(
        collection_name=f"{collection_name}",
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.DOT,
            on_disk=True,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=8,
            indexing_threshold=0,
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True),
        ),
    )


for batch_ids, batch_vectors in tqdm(
    zip(batch_iterate(doc_ids, batch_size), batch_iterate(doc_vectors, batch_size)),
    total=num_rows // batch_size,
    desc="Ingesting in batches",
):
    client.upload_collection(
        collection_name=collection_name,
        vectors=batch_vectors,
        payload=reader.fetch_document_info(pmids=batch_ids),
    )