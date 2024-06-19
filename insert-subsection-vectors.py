import fitz
from sentence_transformers import SentenceTransformer
import time
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

collection_name = "math136_sub_vectors"

connections.connect("default", host="localhost", port="19530")

collection = Collection(name=collection_name)

def find_next_subsection_start(page_text, current_subsection_index, subsection_titles):
    """
    Attempts to find the start index of the next subsection within the page_text.
    Returns the index of the next subsection title in the page_text, or None if not found.
    """
    # Ensure the subsection titles are in the order they appear in the document
    if current_subsection_index + 1 < len(subsection_titles):
        next_subsection_title = subsection_titles[current_subsection_index + 1]
        # Try to find the next subsection title in the page_text
        start_index = page_text.find(next_subsection_title)
        if start_index != -1:
            return start_index
    return None


def extracting_subsections(pdf_path, subsections):
    doc = fitz.open(pdf_path)
    # Convert subsections dictionary to a list of titles to maintain order
    subsection_titles = list(subsections.keys())
    subsection_texts = {title: "" for title in subsection_titles}
    current_subsection_index = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        
        while current_subsection_index < len(subsection_titles):
            # Check if this page contains the start of the current subsection
            if page_num + 1 >= subsections[subsection_titles[current_subsection_index]]:
                # Find the start of the next subsection to determine where to stop appending text
                next_start_index = find_next_subsection_start(page_text, current_subsection_index, subsection_titles)
                if next_start_index is not None:
                    # Append text up to the start of the next subsection
                    subsection_texts[subsection_titles[current_subsection_index]] += page_text[:next_start_index].strip()
                    page_text = page_text[next_start_index:]  # Prepare remaining text for the next loop iteration
                    current_subsection_index += 1  # Move to the next subsection
                else:
                    # If the next subsection isn't on this page, append all text and break to process the next page
                    subsection_texts[subsection_titles[current_subsection_index]] += page_text.strip()
                    break
            else:
                break
    return subsection_texts

def text_to_vectors(chapter_content):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(chapter_content)
    return vectors

# Example usage
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
subsections = {'Vectors in R𝑛': 5, 'Algebraic and Geometric Representation of Vectors': 6, 'Operations on Vectors': 7, 'Vectors in C𝑛': 12, 'Dot Product in R𝑛': 13, 'Projection, Components and Perpendicular': 19, 'Standard Inner Product in C𝑛': 23, 'Fields': 27, 'The Cross Product in R3': 28, 'Linear Combinations and Span': 32, 'Lines in R2': 34, 'Lines in R𝑛': 36, 'The Vector Equation of a Plane in R𝑛': 40, 'Scalar Equation of a Plane in R3': 44, 'Introduction': 48, 'Systems of Linear Equations': 49, 'An Approach to Solving Systems of Linear Equations': 55, 'Solving Systems of Linear Equations Using Matrices': 61, 'The Gauss–Jordan Algorithm for Solving Systems of Linear Equations': 66, 'Rank and Nullity': 75, 'Homogeneous and Non-Homogeneous Systems, Nullspace': 78, 'Solving Systems of Linear Equations over C': 84, 'Matrix–Vector Multiplication': 84, 'Using a Matrix\u2013Vector Product to Express a System of\nLinear Equations': 88, 'Solution Sets to Systems of Linear Equations': 89, 'The Column and Row Spaces of a Matrix': 97, 'Matrix Equality and Multiplication': 99, 'Arithmetic Operations on Matrices': 103, 'Properties of Square Matrices': 107, 'Elementary Matrices': 109, 'Matrix Inverse': 112, 'The Function Determined by a Matrix': 121, 'Linear Transformations': 122, 'The Range of a Linear Transformation and \u201cOnto\u201d Linear Transformations': 127, "The Kernel of a Linear Transformation and \u201cOne-to-\nOne\u201d Linear Transformations": 130, 'Every Linear Transformation is Determined by a Matrix': 133, 'Special Linear Transformations: Projection, Perpendicular, Rotation and Reflection': 136, 'Composition of Linear Transformations': 143, 'The Definition of the Determinant': 147, 'Computing the Determinant in Practice: EROs': 153, 'The Determinant and Invertibility': 156, 'An Expression for 𝐴−1': 158, "Cramer’s Rule": 161, 'The Determinant and Geometry': 163, 'What is an Eigenpair?': 167, 'The Characteristic Polynomial and Finding Eigenvalues': 169, 'Properties of the Characteristic Polynomial': 172, 'Finding Eigenvectors': 178, 'Eigenspaces': 181, 'Diagonalization': 183, 'Subspaces': 190, 'Linear Dependence and the Notion of a Basis of a Subspace': 193, 'Detecting Linear Dependence and Independence': 197, 'Spanning Sets': 204, 'Basis': 208, 'Bases for Col(𝐴) and Null(𝐴)': 212, 'Dimension': 216, 'Coordinates': 219, 'Matrix Representation of a Linear Operator': 230, 'Diagonalizability of Linear Operators': 235, 'Diagonalizability of Matrices Revisited': 238, 'The Diagonalizability Test': 244, 'Definition of a Vector Space': 254, 'Span, Linear Independence and Basis': 258, 'Linear Operators': 264, 'Index': 270}
subsection_content_dict = extracting_subsections(pdf_path, subsections)
del subsection_content_dict['Index']

subsection_headers = list(subsection_content_dict.keys())
subsection_content = list(subsection_content_dict.values())

for i in subsection_content:
    print(i)

vectors = text_to_vectors(subsection_content)

# Insert vectors into collection
batch_size = 100  # You can adjust this based on your data size and system resources
start_time = time.time()
num_batches = len(vectors) // batch_size
for i in range(num_batches):
    vectors_batch = vectors[i * batch_size: (i + 1) * batch_size]
    subsection_content_batch = subsection_content[i * batch_size: (i + 1) * batch_size] #NEWLINE
    subsection_headers_batch = subsection_headers[i * batch_size: (i + 1) * batch_size]
    entities = [
        {
            "text_vector": vector.tolist(),
            "subsection_content": content,
            "subsection_headers" : header
        }
        for vector, content, header in zip(vectors_batch, subsection_content_batch, subsection_headers_batch)
    ]
    collection.insert(entities)

# Insert the remaining vectors if the total number is not divisible by batch_size
remaining_vectors = len(vectors) % batch_size
if remaining_vectors > 0:
    entities = [
        {
            "text_vector": vectors[-remaining_vectors:][j].tolist(),
            "subsection_content": subsection_content[-remaining_vectors:][j],
            "subsection_headers" : subsection_headers[-remaining_vectors:][j]
        }
        for j in range(remaining_vectors)
    ]
    collection.insert(entities)

print(f"Data insertion completed in {time.time() - start_time} seconds.")

import json
with open('math136-sub-content', 'w') as fout:
    json.dump(subsection_content, fout)
with open("math136-sub-vectors",'w') as fout:
    vectors_list = vectors.tolist()
    json.dump(vectors_list, fout)
with open("math136-sub-headers",'w') as fout:
    json.dump(subsection_headers, fout)
