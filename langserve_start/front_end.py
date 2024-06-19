#FIX CHAT FLOW FOR CONTINUOUS CHAT CONVOS

import streamlit as st
import requests

FASTAPI_SERVER_URL = "http://localhost:8000/ask"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title('LangChain Question Answering Interface')

question = st.text_input("Enter your question:")

if st.button("Submit"):
    if question:
        response = requests.post(
            FASTAPI_SERVER_URL,
            json={
                "question": question,
                "chat_history": st.session_state.chat_history
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.chat_history.append({'question': question, 'answer': data['answer']})
            for chat in st.session_state.chat_history:
                st.write(f"Q: {chat['question']}")
                st.write(f"A: {chat['answer']}")
                if 'details' in chat and chat['details']:
                    st.write("Details:")
                    st.json(chat['details'])
        else:
            st.error("Failed to get an answer from the server.")
    else:
        st.warning("Please enter a question.")