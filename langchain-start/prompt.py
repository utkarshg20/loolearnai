from langchain_openai import ChatOpenAI

OPENAI_API_KEY="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"

#GENERATE NORMAL OUTPUT
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

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
print(chain.invoke({"user_input": "how can langsmith help with testing?"}))