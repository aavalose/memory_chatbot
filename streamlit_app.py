import streamlit as st
import torch
from transformers import pipeline
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder)
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

# Initialize session state for memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

def get_ml_response(query):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    
    # Initialize pipeline if not in session state
    if "pipe" not in st.session_state:
        st.session_state.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device=device,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            truncation=True,
        )
    
    # Wrap the pipeline in HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=st.session_state.pipe)

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a chatbot that is an expert on machine learning."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Create LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

    # Invoke the chain with the query
    output = llm_chain.invoke({"input": query})
    
    # Save conversation to file
    with open('conversation_history.txt', 'a') as f:
        f.write(f"User: {query}\n")
        f.write(f"Assistant: {output['text']}\n\n")
    
    return output['text']

# Streamlit UI
st.title("ML Expert Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about machine learning?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_ml_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add a button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True) 