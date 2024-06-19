import time
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import fitz

#collection_name = "textbook_vectors"
collection_name = "math136_chap_vectors"

connections.connect("default", host="localhost", port="19530")

collection = Collection(name=collection_name)

def text_to_vectors(chapter_content):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(chapter_content)
    return vectors

def extract_text_by_chapter(pdf_path):
    doc = fitz.open(pdf_path)
    chapters = {"Pre-Chapter": ''}  # Dictionary to store chapter texts
    chapter_title = "Pre-Chapter"  # Default chapter title for content before the first chapter

    for page_num in range(len(doc)):
        page_text = doc.load_page(page_num).get_text()
        
        # Attempt to identify chapter starts and titles
        if "Chapter" in page_text.split('\n')[0]:  # Assuming chapters start at the beginning of pages
            chapter_title = page_text.split('\n')[0]  # Simplified assumption
            chapters[chapter_title] = ""  # Initialize chapter text
        
        chapters[chapter_title] += page_text  # Append text to current chapter

    return chapters

chapter_content = []
#page_content = []
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
chapters = extract_text_by_chapter(pdf_path)
del chapters['Pre-Chapter']
chapter_headers = list(chapters.keys()) 
chapter_content = list(chapters.values())
for i in chapter_content:
    print(len(i))
vectors = text_to_vectors(chapter_content)
# Insert vectors into collection
batch_size = 120  # You can adjust this based on your data size and system resources
start_time = time.time()
num_batches = len(vectors) // batch_size
for i in range(num_batches):
    vectors_batch = vectors[i * batch_size: (i + 1) * batch_size]
    chapter_content_batch = chapter_content[i * batch_size: (i + 1) * batch_size] #NEWLINE
    chapter_headers_batch = chapter_headers[i * batch_size: (i + 1) * batch_size]
    entities = [
        {
            "text_vector": vector.tolist(),
            "chapter_content": content,
            "chapter_header" : header
        }
        for vector, content, header in zip(vectors_batch, chapter_content_batch, chapter_headers_batch)
    ]
    collection.insert(entities)

# Insert the remaining vectors if the total number is not divisible by batch_size
remaining_vectors = len(vectors) % batch_size
if remaining_vectors > 0:
    '''entities = [
        {"text_vector": vector.tolist()}  # Convert to list for insertion
        for vector in vectors[-remaining_vectors:]
    ]'''
    entities = [
        {
            "text_vector": vectors[-remaining_vectors:][j].tolist(),
            "chapter_content": chapter_content[-remaining_vectors:][j],
            "chapter_header" : chapter_headers[-remaining_vectors:][j]
        }
        for j in range(remaining_vectors)
    ]
    collection.insert(entities)

print(f"Data insertion completed in {time.time() - start_time} seconds.")

import json
with open('math136-chap-text', 'w') as fout:
    json.dump(chapter_content, fout)
with open("math136-chap-vectors",'w') as fout:
    vectors_list = vectors.tolist()
    json.dump(vectors_list, fout)
with open("math136-chap-headers",'w') as fout:
    json.dump(chapter_headers, fout)


