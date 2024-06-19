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

client = OpenAI(api_key="")

model = SentenceTransformer('all-MiniLM-L6-v2')

question = "Prove that if two planes interesect in R3 then the solution is either a line or a plane"

collection_name = "math136_vectors"
connections.connect("default", host="localhost", port="19530")
collection = Collection(name=collection_name)
collection.load()

pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
doc = fitz.open(pdf_path)
toc = doc.get_toc()
doc.close()

subsection = {}
for i in toc:
    if i[0]==2:
        subsection[i[1]]=i[2]

new_toc = []
start_appending = 0
subsection_start = "Algebraic and Geometric Representation of Vectors"
subsection_threshold = "Solution Sets to Systems of Linear Equations"
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
subsection_encoded = []
ini = 100
for i in new_toc:
    subsection_encoded.append((str(ini) + i))
    ini+=1

sub_query = '''The question is as follows: {}.\n Based on the provided subsections, please identify the specific topics necessary to solve this proof using core concepts of linear algebra. It's crucial to use the exact names as listed below. If a similar concept is needed but not listed, please refer to the closest listed equivalent. STRICTLY USED THE ENCODED SUBSECTION NAMES PROVIDED IN THE LIST BELOW.\nListed Subsections: {}\n Moreover, please list the steps required for you to solve the question. Please mention "Steps to solve the question:" when you start explaining the steps. Do not start solving the question.'''.format(question, subsection_encoded)

topics_steps = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": "You are a mathematical professor skilled in explaining and solving linear algebra questions algebraically. Provide clear, step-by-step algebraic explanations. You teach introduction to linear algebra course."},
    {"role": "user", "content": sub_query},
  ],
  max_tokens=400, # Adjust based on how long you expect the answer to be
  temperature=0.0, # A higher temperature encourages creativity. Adjust based on your needs
  top_p=0.0,
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
for i in subsection_encoded:
    if i[:3] in selected_subsections:
        selected_subheaders[i] = 'TRUE'

print(selected_subheaders, subsection_encoded)
print(output[1])