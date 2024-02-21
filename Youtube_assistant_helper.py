from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

def create_vector_db_from_youtube_url(video_url:str)-> Chroma:
    loader= YoutubeLoader.from_youtube_url(video_url)
    transcript= loader.load()
    
    text_splitter= RecursiveCharacterTextSplitter()
    docs= text_splitter.split_documents(transcript)
    
    db= Chroma.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query,k=25):
    #the mixtralmodel can handle 32k tokens to define k max= nbrtokens/chunk_size
    
    docs= db.similarity_search(query, k=k)
    docs_page_content= " ".join([d.page_content for d in docs])
    
    llm= HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.2})
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response=chain.run(question=query, docs=docs_page_content)
    #response=chain.invoke({'question': query, 'docs': docs_page_content})
    response=response.replace("\n"," ")
    return response