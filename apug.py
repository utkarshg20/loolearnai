from openai import OpenAI
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
import fitz
import re

#ADD ENCODING TO AVOID PREVIOUS ISSUE
client = OpenAI(api_key="=")

model = SentenceTransformer('all-MiniLM-L6-v2')

#question = "Prove that if two planes interesect in R3 then the solution is either a line or a plane"
#question = "Explain the Fundamental Theorem of Calculus (Part 1)"
question = '''one-sided inverse'''

collection_name = "math136_vectors"
connections.connect("default", host="localhost", port="19530")
collection = Collection(name=collection_name)
#collection.create_index(field_name="text_vector", index_params={"index_type": "IVF_SQ8", "metric_type": "L2", "nlist": 16384})
collection.load()

pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH138-notes.pdf'
doc = fitz.open(pdf_path)
toc = doc.get_toc()
doc.close()

chapter_num = 0
subsection_num = 0
subsection = {}
for i in toc:
    if i[0]==2 or i[0]==3:
        subsection_num += 1
        subsection["{}.{} ".format(chapter_num, subsection_num) + i[1]]= i[2]
    elif i[0]==1:
        subsection_num = 0
        chapter_num += 1
new_toc = []
print(subsection.keys())
start_appending = 0
subsection_start = "Operations on Vectors"
subsection_threshold = "The Determinant"
for i in toc:
    if i[0] == 2 or i[0]==3:
        if start_appending > 0:
            new_toc.append(i[1])
        if i[1]==subsection_start:
            start_appending+=1
            new_toc.append(i[1])
        if i[1]==subsection_threshold:
            new_toc.append(i[1])
            break
print(new_toc)
sub_query = '''The question is as follows: {}.\n Based on the provided subsections, please identify the specific topics necessary to solve this proof using core concepts of linear algebra. It's crucial to use the exact names as listed below. If a similar concept is needed but not listed, please refer to the closest listed equivalent.STRICTLY USED THE ENCODED SUBSECTION NAMES PROVIDED IN THE LIST BELOW.\nListed Subsections: {}\n Moreover, please list the steps required for you to solve the question. Please mention "Steps to solve the question:" when you start explaining the steps. Do not start solving the question.'''.format(question, new_toc)
#sub_query = '''The question is as follows: {}.\n Based on the provided subsections, please identify the specific topics necessary to solve this proof using core concepts of calculus. It's crucial to use the exact names as listed below. If a similar concept is needed but not listed, please refer to the closest listed equivalent.STRICTLY USED THE ENCODED SUBSECTION NAMES PROVIDED IN THE LIST BELOW.\nListed Subsections: {}\n Moreover, please list the steps required for you to solve the question. Please mention "Steps to solve the question:" when you start explaining the steps. Do not start solving the question.'''.format(question, new_toc)

topics_steps = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": "You are a mathematical professor skilled in explaining and solving linear algebra questions algebraically. Provide clear, step-by-step algebraic explanations. You teach introduction to linear algebra course."},
    #{"role": "system", "content": "You are a mathematical professor skilled in explaining and solving calculus questions. Provide clear, step-by-step mathematical explanations with steps and theorems cited. You teach introduction to Calculus 2 course."},
    {"role": "user", "content": sub_query},
  ],
  max_tokens=400, # Adjust based on how long you expect the answer to be
  temperature=0, # A higher temperature encourages creativity. Adjust based on your needs
  top_p=0,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
)

output = str(topics_steps.choices[0].message)
output = output.split("Steps to solve the question:\\n")
print(output, "\n\n")

selected_subsections = output[0]
replace = str("\\\( \\\mathbb{")
selected_subsections = selected_subsections.replace(replace, '')
selected_subsections = selected_subsections.replace('}^', '')
intro = 0
selected_subheaders = {}
for i in new_toc:
    if i in selected_subsections:
        selected_subheaders[i] = subsection[i]

print(selected_subheaders, new_toc)

steps = output[1]
steps_list = re.split(r'(?=\d+\.)', steps)
steps_list = steps_list[1:]
print(steps_list)

sorted_subsections = sorted(subsection.items(), key=lambda item: item[1])
# Find the page ranges for selected subsections
selected_pages = []
for title, page in selected_subheaders.items():
    start_page = page
    end_page = None
    start_index = sorted_subsections.index((title, page))
    
    # Look for the next subsection's start page to determine the end page of the current subsection
    if start_index + 1 < len(sorted_subsections):
        end_page = sorted_subsections[start_index + 1][1]
    
    selected_pages.append((start_page, end_page))

expr = ""
for i in selected_pages:
    if expr=="":
        expr += "(page_numbers >= {} && page_numbers <= {})".format(i[0], i[1])
    else: 
        expr += " || (page_numbers >= {} && page_numbers <= {})".format(i[0], i[1])

extracted_content = 'Consider the following points extracted from the textbook for a detailed explanation:\n'
extracted_pagenums = ''

if len(steps_list) >= 6:
  top_k = 1
elif len(steps_list) >=3:
  top_k = 2
else:
  top_k = 3

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 12},
}

print(expr)

for i in steps_list:
  # Convert texts to vectors
  query_vector = model.encode([i])[0].tolist()
  # Perform a vector similarity search
  query_result = collection.search([query_vector], "text_vector", search_params, limit = top_k, output_fields=["text_content", "page_numbers"], expr = expr)
  for res in query_result:
    for entity in res:
        extracted_content += (entity.text_content + "\n\n")
        extracted_pagenums += str(entity.page_numbers) + "\n"

#query = "Question to solve using core concepts of calculus: {} \n Context from book: {} \nINSTRUCTIONS: For 'prove or disprove' or 'true or false' questions please consider all trivial and edge cases. For proofs provide rigourous calculus proofs based on pre-existing and content provided".format(question, extracted_content)
query = "Question to solve algebraically: {} \n Context from book: {} \nINSTRUCTIONS: For 'prove or disprove' or 'true or false' questions please consider all trivial and edge cases. For proofs provide rigourous algebric proofs based on pre-existing and content provided".format(question, extracted_content)
#query = corrected_query = str(TextBlob(query).correct()) #USE TO CORRECT TYPOS BUT RUINS IT FOR MATHEMATICAL THINGS
#sub_query = '''The question is as follows: {}.\n Please identify which of the following specific subsections are necessary for this proof to be solved using core concepts of linear algebra. It's important to use the exact names as listed:{}'''.format(query, str(new_toc))

final_answer = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
      {"role": "system", "content": "You are a mathematical professor at university of waterloo, skilled in explaining and solving linear algebra questions completely algebraically"},
      #{"role": "system", "content": "You are a mathematical professor skilled in explaining and solving calculus questions. Provide clear, step-by-step mathematical explanations with steps and theorems cited. You teach introduction to Calculus 2 course."},
      {"role": "user", "content": query}
    ],
  max_tokens=500, # Adjust based on how long you expect the answer to be
  temperature=0.1, # A higher temperature encourages creativity. Adjust based on your needs
  top_p=0.5,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None # You can specify a stop sequence if there's a clear endpoint. Otherwise, leave it as None
  #stream = True
)

print(final_answer)
print(extracted_pagenums)