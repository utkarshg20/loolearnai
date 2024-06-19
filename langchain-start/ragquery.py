from langchain_openai import ChatOpenAI


#GENERATE NORMAL OUTPUT
llm = ChatOpenAI()
print(llm.invoke("how can langsmith help with testing?"))
print(llm.invoke("what is the difference between user and human in langchain chat prompt template"))

#PROMPT TEMPLPATE
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    #("human", "You are a world class technical documentation writer."),
    #("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = prompt.invoke(
    {
        "name": "Bob",
        "user_input": "how can langsmith help with testing?"
    }
)

#MAKE A CHAIN
chain = prompt | llm 

#CONVERT OUTPUT TO STRING
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

#NEW CHAIN
chain = prompt | llm | output_parser

#MAKING A RETRIEVAL CHAIN (RAG BASICALLY)
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("./apug data/MATH136-notes.pdf")
docs = loader.load()
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(["https://docs.smith.langchain.com/user_guide"])


#MAKING EMBEDDINGS
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

#SPLITTING TEXT INTO CHUNKS AND STORING EMBEDDINGS (VECTORS) 
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

#Set up for a chain that takes a question and the retrieved documents and generates an answer
from langchain.chains.combine_documents import create_stuff_documents_chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

from langchain_core.documents import Document
document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

#DYNAMICALLY PICK RELEVANT PARTS OF THE DOCUMENT AND PASS IT
from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])