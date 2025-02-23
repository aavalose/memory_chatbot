import torch
from transformers import pipeline
from langchain_core.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder)
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

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

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a chatbot that is an expert on machine learning."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Create LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # Invoke the chain with the query
    output = llm_chain.invoke({"input": query})
    
    # Save conversation to file
    with open('conversation_history.txt', 'a') as f:
        f.write(f"User: {query}\n")
        f.write(f"Assistant: {output['text']}\n\n")
    
    return output['text']

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Example usage
response = get_ml_response("Hi, how are you?", memory)
print(response)

response = get_ml_response("What did I just ask you?", memory)
print(response)
