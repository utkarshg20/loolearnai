import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

connections.connect("default", host="localhost", port="19530")


collection_name = "math138_vectors"

dim = 384
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim = dim),
#    FieldSchema(name="subsection_content", dtype=DataType.VARCHAR, max_length = 30000), 
#    FieldSchema(name="subsection_headers", dtype=DataType.VARCHAR, max_length = 400)
    FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length = 4500), 
    FieldSchema(name="page_numbers", dtype=DataType.INT64)
]

schema = CollectionSchema(fields=fields, description="MATH138 Textbook Chapter Vectors Collection")
collection = Collection(name=collection_name, schema=schema)
print(f"Collection {collection_name} created.")