import streamlit as st
from langchain_community.chat_models import ChatCohere
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()
st.title("Langchain demo")
cohere_api_key=os.environ.get("COHERE_API_KEY")
def generate_response(input_text):
    llm=ChatCohere(
        temperature=0.5,
        max_tokens=256,
        model="command",
        cohere_api_key=cohere_api_key
        )
    messages=[HumanMessage(content=input_text)]
    st.info(llm.invoke(messages).content)
with st.form("chat_form"):
    text=st.text_area("Enter Text","What is the import")
    submitted = st.form_submit_button("submit")
    if submitted:
        generate_response(text)
