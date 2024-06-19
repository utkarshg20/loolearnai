from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter 
loader = TextLoader("./apug data/MATH136-notes.pdf")
loader.load()