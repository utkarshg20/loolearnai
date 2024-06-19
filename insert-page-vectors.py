import time
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

#collection_name = "textbook_vectors"
collection_name = "math138_vectors"

connections.connect("default", host="localhost", port="19530")

collection = Collection(name=collection_name)
#collection.drop()
#collection.delete(expr="page_numbers > -100")
import fitz

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

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    page_info = {}
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        page_info[page_num + 1] = page_text

    return page_info

def text_to_vectors(chapter_content):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(chapter_content)
    return vectors

#chapter_content = []
page_content = []
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH138-notes.pdf'
#chapters = extract_text_by_chapter(pdf_path)
pages = extract_text_by_page(pdf_path)
#for i in chapters:
for i in pages:
    if i!="Pre-Chapter":
        page_content.append(pages[i])
        #chapter_content.append(chapters[i])

#vectors = text_to_vectors(chapter_content)
vectors = text_to_vectors(page_content)
page_numbers = list(pages.keys())
# Insert vectors into collection
batch_size = 100  # You can adjust this based on your data size and system resources
start_time = time.time()
num_batches = len(vectors) // batch_size
for i in range(num_batches):
    vectors_batch = vectors[i * batch_size: (i + 1) * batch_size]
    page_content_batch = page_content[i * batch_size: (i + 1) * batch_size] #NEWLINE
    '''entities = [
        {"text_vector": vector.tolist()}  # Convert to list for insertion
        for vector in vectors_batch
    ]'''
    entities = [
        {
            "text_vector": vector.tolist(),
            "text_content": content,
            "page_numbers" : number
        }
        for vector, content, number in zip(vectors_batch, page_content_batch, page_numbers)
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
            "text_content": page_content[-remaining_vectors:][j],
            "page_numbers" : page_numbers[-remaining_vectors:][j]
        }
        for j in range(remaining_vectors)
    ]
    collection.insert(entities)

print(f"Data insertion completed in {time.time() - start_time} seconds.")

import json
with open('math138-text', 'w') as fout:
    json.dump(pages, fout)
with open("math138-vectors",'w') as fout:
    vectors_list = vectors.tolist()
    json.dump(vectors_list, fout)
with open("math138-page_numbers",'w') as fout:
    json.dump(page_numbers, fout)


