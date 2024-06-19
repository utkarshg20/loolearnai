from langchain_openai import ChatOpenAI

OPENAI_API_KEY=""

#GENERATE NORMAL OUTPUT
llm = ChatOpenAI(api_key=OPENAI_API_KEY)
print(llm.invoke("how can langsmith help with testing?"))
print(llm.invoke("what is the difference between user and human in langchain chat prompt template"))

'''
Langsmith can help with testing by providing automated testing tools and frameworks that can streamline the testing process. 
It can also help with creating test cases, running tests, and analyzing the results to identify any issues or bugs in the software.
Additionally, Langsmith can assist with performance testing, security testing, and regression testing to ensure the software meets 
quality standards before deployment.
'''

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
#print(chain.invoke({"user_input": "how can langsmith help with testing?"}))

#CONVERT OUTPUT TO STRING
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

#NEW CHAIN
chain = prompt | llm | output_parser
#print(chain.invoke({"user_input": "how can langsmith help with testing?"}))

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

#AGENT

#CREATING TOOLS FOR THE AGENT
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.python import PythonREPL
#PERFORMS WEB SEARCH FOR QUERY (CONNECT LLM TO THE WEB)
TAVILY_API_KEY = ""
search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
tools = [retriever_tool, search]

#BUILDING THE AGENT
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# You need to set OPENAI_API_KEY environment variable or pass it as argument `api_key`.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key = OPENAI_API_KEY)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "what is the weather in SF?"})