from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
import openai
import os
import streamlit as st

# Ensure the OpenAI API key is set
openai.api_key = "input your openai key"
# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
if 'ai-assistant' not in pc.list_indexes().names():
    pc.create_index(
        name='ai-assistant',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws',region='us-west-2')
        
    )
index = pc.Index('ai-assistant')

# Function to find matching responses
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

# Function to refine the user's query based on the conversation log
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

# Function to get the conversation string
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
