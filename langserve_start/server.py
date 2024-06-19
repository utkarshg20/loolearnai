#!/usr/bin/env python
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.messages import BaseMessage

# API keys
OPENAI_API_KEY=""
TAVILY_API_KEY = ""

# 1. Load documents and prepare the retriever
loader = PyPDFLoader("C:\\Users\\Utki\\Desktop\\code\\project\\apug data\\MATH136-notes.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Create tools for the agent
retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about LangSmith.")
search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
tavily_tool = TavilySearchResults(api_wrapper=search)
tools = [retriever_tool, tavily_tool]

# 3. Create the agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Define FastAPI application
app = FastAPI(title="LangChain Server", version="1.0", description="A simple API server using LangChain's Runnable interfaces")

# Define the input/output data models
class Query(BaseModel):
    question: str
    chat_history: list

class Answer(BaseModel):
    answer: str
    chat_history: list

# 5. Add a route to handle questions
@app.post("/ask", response_model=Answer)
async def ask_question(query: Query):
    try:
        # This is where you invoke the agent with the user's question
        answer = f"Echo: {query.question}"
        new_chat = query.chat_history + [{'question': query.question, 'answer': answer}]
        return Answer(answer=answer, chat_history=new_chat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
