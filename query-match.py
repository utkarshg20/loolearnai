from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from textblob import TextBlob

model = SentenceTransformer('all-MiniLM-L6-v2')
# Convert texts to vectors
query = '''AmatrixA∈Mm×n(R) issaidtohavefull rankif rank(A)=min{m,n}. Forsuch matrices,wecandefineanappropriateone-sidedinverse:'''
#query = corrected_query = str(TextBlob(query).correct()) #USE TO CORRECT TYPOS BUT RUINS IT FOR MATHEMATICAL THINGS
query_vector = model.encode([query])[0].tolist()
# NOTE: QUERY AND CONTENT VECTOR ARE SAME DIMENSION SO ITS GOOD.


#collection_name = "textbook_vectors"
collection_name = "math136_vectors"
connections.connect("default", host="localhost", port="19530")
collection = Collection(name=collection_name)
#collection.create_index(field_name="text_vector", index_params={"index_type": "IVF_SQ8", "metric_type": "L2", "nlist": 16384})
#collection.drop()
collection.load()

schema = collection.schema

print("Collection Schema:", schema)

top_k = 3
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 20},
}

expr = "(page_numbers >= 7 && page_numbers <= 8) || (page_numbers >= 5 && page_numbers <= 6)"
expr = "(page_numbers >= 109 && page_numbers <= 112) || (page_numbers >= 113 && page_numbers <= 115)"
#expr = "(page_numbers >= 109 && page_numbers <= 112) || (page_numbers >= 112 && page_numbers <= 121)"
# Perform a vector similarity search
#query_result = collection.search([query_vector], "text_vector", search_params, limit=5, output_fields=["text_vector", "subsection_content", "subsection_headers"])
#results = collection.query(expr='page_numbers >= 109 && page_numbers <= 112', output_fields=['text_content', 'page_numbers'])
#for result in results:
#    print(f"Page Number: {result['page_numbers']}, Text Content: {result['text_content'][:100]}...")
print("POOOOOOOOOOOOO")
query_result = collection.search([query_vector], "text_vector", search_params, limit=5, output_fields=["text_vector", "text_content", "page_numbers"]) #expr = expr
#query_result = collection.search([query_vector], "text_vector", search_params, limit=3, output_fields=["random"])
#query_result = collection.query([{"text_vector": query_vector}], top_k=top_k)

# Process and print the query result
for res in query_result:
    for entity in res:
        vector_id = entity.id
        # Print the vector ID, vector, and distance
        print("Vector ID:", vector_id)
    #    print("Vector:", entity.text_vector)
    #    print("Subsection Content:", entity.subsection_content)
        print("Page Content:", entity.text_content)
    #    print("Subsection Header:", entity.subsection_headers)
        print("Page Number:", entity.page_numbers)
        print("Distance:", entity.distance)
print(query)

import fitz
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
doc = fitz.open(pdf_path)
toc = doc.get_toc()
doc.close()
print(toc) #75 #70 NUMBER OF METADATA CONTENTS