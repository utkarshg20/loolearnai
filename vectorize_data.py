'''

# APPROACH 1...........................................

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Example text
texts = ["This is an example sentence", "Here's another one"]

# Convert texts to vectors
embeddings = model.encode(texts)

print(embeddings)'''

'''
# APPROACH 2...........................................
import fitz  # PyMuPDF

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
pdf_path = r'C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf'
chapters = extract_text_by_chapter(pdf_path)
for i in chapters:
    if i!="Pre-Chapter":
        chapter_content.append(chapters[i])

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert texts to vectors
embeddings = model.encode(chapter_content)

for i in embeddings:
    print(len(i))
'''
query = ["What is 1 ≡ x mod 10?",]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
# Convert texts to vectors
embeddings = model.encode(query)
for i in embeddings:
    print(len(i))

# Example: Textbook content as a list of strings (you would replace this with the actual content vectors)
textbook_content = ["Chapter 1: Introduction to Congruence", "Chapter 2: Modern Trigonometry"]
content_vecs = model.encode(textbook_content)

# Calculate similarity scores
similarity_scores = cosine_similarity(embeddings, content_vecs)

# Print the scores or process them to find the most relevant chapter
print(similarity_scores)

#WORKS INSPITE OF MOD IN MODERN AND CONGRUENCE IS SAME AS MOD