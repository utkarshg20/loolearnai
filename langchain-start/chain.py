from langchain_openai import ChatOpenAI


#GENERATE NORMAL OUTPUT
llm = ChatOpenAI()
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
