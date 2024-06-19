import time
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

collection_name = "math136_vectors"

connections.connect("default", host="localhost", port="19530")

collection = Collection(name=collection_name)
# Get the number of entities (vectors) in the collection
num_entities = collection.num_entities
# Print the number of vectors
print("Number of vectors in the collection:", num_entities)