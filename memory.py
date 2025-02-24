import torch
import sys
import os
from transformers import pipeline
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder)
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from embed import embed_mini, get_top_k_similarities
from qa_dict_gen import qa_dict_gen


def get_ml_response(query, memory):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
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
    llm = HuggingFacePipeline(pipeline=pipe)

    # Embed the query and find most similar from embeddings file
    query_embedding = embed_mini(query)
    saved_embeddings = torch.load('arraigo_embeddings.pt', weights_only=False)

    qa_dict = qa_dict_gen('qa_arraigo')
    questions = list(qa_dict.keys())
    
    # Get most similar embedding index and document
    similar_idx = get_top_k_similarities(query_embedding, saved_embeddings, k=1)[0]
    similar_doc = questions[similar_idx]
    similar_answer = qa_dict[similar_doc]

    # Define prompt template with similar embedding context and answer
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a chatbot that is an expert on Migration to Spain under the Arraigo program."),
        MessagesPlaceholder(variable_name="history"),
        SystemMessage(content=f"Here is some relevant context: Question: {similar_doc}\nAnswer: {similar_answer}"),
        SystemMessage(content=f"The user asks: {query}")
    ])

    # Create LLM chain with memory
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Use load_memory_variables to get the history
    history = memory.load_memory_variables({}).get("history", [])

    # Invoke the chain with the query and history
    output = llm_chain.invoke({"input": query, "history": history})
    
    # Save conversation to file
    with open('conversation_history.txt', 'a') as f:
        f.write(f"User: {query}\n")
        f.write(f"{output['text']}\n\n")
    
    # Extract just the final answer from the output
    response = output['text'].split("Answer:")[-1].strip()
    return response
