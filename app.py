import streamlit as st
import numpy as np
import random
import time
import cohere
import os
from dotenv import load_dotenv
load_dotenv()
from cohere.responses.chat import StreamEvent 
# with st.chat_message("user"):
#     st.write("hello")
#     st.bar_chart(np.random.randn(30,3))
co=cohere.Client(os.environ.get("COHERE_API_KEY"))
st.title("Chat bot")
# def response_generator():
#     response=[
#         "hello there! how can i help you",
#         "I im here to help. what do you need?",
#         "Hey! what can i do for you?",
#         "Hi! what do you need help with?"
#     ]
#     response=random.choice(response)
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)
def cohere_response_generator(prompt):
    chat_history=list(map(lambda x:{
        "user_name":"user" if x["role"]=="user" else "Chatbot",
        "text":x["content"]
    },st.session_state.messages))
    for event in co.chat(f'{prompt}.Apply common sense. Answer in less than 500 words.',chat_history=chat_history,stream=True):
        if event.event_type==StreamEvent.TEXT_GENERATION:
            yield event.text
        elif event.event_type==StreamEvent.STREAM_END:
            return ""
# initiaize a chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# display the chat messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("say something.."):
    #display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    #Add the user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    #Display the assistant response in chat message container
    with st.chat_message("assistant"):
        # st.markdown(response)
        # response=st.write_stream(response_generator())
        # stream=co.chat(prompt,streaming=True)
        response=st.write_stream(cohere_response_generator(prompt))

    #Add the assistant response to the chat history
    st.session_state.messages.append({"role":"assistant","content":response})

