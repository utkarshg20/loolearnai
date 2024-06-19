from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("./apug data/MATH136-notes.pdf")
docs = loader.load()
#MAKING EMBEDDINGS
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#SPLITTING TEXT INTO CHUNKS AND STORING EMBEDDINGS (VECTORS) 
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
from langchain.chains import create_retrieval_chain
retriever = vector.as_retriever()

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