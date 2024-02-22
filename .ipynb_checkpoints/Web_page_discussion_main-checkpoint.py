import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_vectorstore_from_URL(url):
    # get the text in doc format
    loader= WebBaseLoader(url)
    documents=loader.load()
    
    #split documents
    text_splitter= RecursiveCharacterTextSplitter()
    chunks_docs= text_splitter.split_documents(documents)

    #create the vectorstore
    embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    vec_store= Chroma.from_documents(chunks_docs, embeddings)
    return vec_store
    
def get_context_retriever_chain(vectorstore):
    llm= HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5})
    retriever= vectorstore.as_retriever()
    prompt= ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="Chat_history"),
        ("human","{input}"),
        ("human","Given the above conversation, generate a search query to look up in order to get informations relevant to the conversation."),
    ])
    
    retriever_chain= create_history_aware_retriever(llm,retriever,prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm= HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5})
    prompt= ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="Chat_history"),
        ("user","{input}"),
    ])
    stuff_documents_chain=create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain,stuff_documents_chain)

def extract_response(text):
    index = text.rfind("AI:")
    if index != -1:
        return text[index + len("AI:"):]
    else:
        return "No resonse after 'AI:'"
    
def get_response(user_in):   
    retriever_chain= get_context_retriever_chain(st.session_state.documents)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response=conversation_rag_chain.invoke({
        "Chat_history": st.session_state.Chat_history,
        "input":user_query
        })
    
    all_text = response["answer"]
    extract_response = extract_response(all_text)
    
    return extract_response

# Configuration
st.set_page_config(page_title="Chat&Draw with AI")
st.title("Chat&Draw with AI")
if "Chat_history" not in st.session_state:
    st.session_state.Chat_history=[AIMessage(content="Hello, I am a Chatbot. Let's chat together!!"),]

# Sidebare
with st.sidebar:
    st.header("Settings")
    web_url=st.text_input("URL")
    
if web_url is None or web_url=="":
    st.info("Please enter an URL")
    
else:
    #session state initialisation
    if "Chat_history" not in st.session_state:
        st.session_state.Chat_history=[
        AIMessage(content="Hello, How can I help you?")
        ]
    if "documents" not in st.session_state:
        st.session_state.documents= get_vectorstore_from_URL(web_url)

    # Inputs from user
    user_query=st.chat_input("Type tour message here...")
    if user_query is not None and user_query!="":
        response= get_response(user_query)
        st.session_state.Chat_history.append(HumanMessage(content=user_query))
        st.session_state.Chat_history.append(AIMessage(content=response))
        
    # Conversation
    for message in st.session_state.Chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    