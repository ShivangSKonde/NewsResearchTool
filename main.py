import pickle
from langchain import OpenAI
import os
import streamlit as st
import time

#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from unstructured.partition.html import partition_html
from sentence_transformers import SentenceTransformer
import faiss

#from langchain_community.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv


"""
  main functions to keep in mind during the interview
  WebBaseLoader(urls) = To load the content of url
  RecursiveCharacterTextSplitter() = to split the text at various separators it divides the whole data into various chunks
  HuggingFaceEmbeddings() = this create the vector embeddings of our various chunks
  RetrievalQAWithSourcesChain() = this retrevies the data from the vectorDB object created using faiss
                                  retreives only relevant chunks and performs map reduce operation
"""

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Articles URLs")

urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")  # URL {i+1} this is just line above the input box
    urls.append(url)

process_url_button_clicked =st.sidebar.button("Process URLs")
file_path="faiss_store_openai.pkl"

main_placeholder=st.empty()
llm=ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.9,
    max_tokens=500,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "News Research Tool"
    }
)


if process_url_button_clicked:
    # load data
    loader=WebBaseLoader(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data=loader.load()

    #Split data
    splitter=RecursiveCharacterTextSplitter(
        separators=['.','\n','\n\n',','],
        chunk_size=800,
        chunk_overlap=100
    )

    docs=splitter.split_documents(data)

    #Store it in vector database after converting it into the vector
    embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_openai=FAISS.from_documents(docs,embeddings)
    main_placeholder.text('Embedding Vector Started Building...✅✅✅')
    time.sleep(2)

    #store the vectorDB object in pickle file
    with open(file_path,'wb') as f:
        pickle.dump(vectorstore_openai,f)


query=main_placeholder.text_input("Question :")

if query:
    #processing the query with the help of vectorDB object
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vectorstore=pickle.load(f)
            # retreving the data from FAISS using the MAp reduce method

            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever = vectorstore.as_retriever(search_kwargs={"k": 6}))
            result=chain({'question':query},return_only_outputs=True)

            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result['answer'])

            # display the sources if available
            sources=result['sources']
            if sources:
                st.subheader("Sources:")
                source_list=sources.split('\n')  # split the sources by new line

                for source in source_list:
                    st.write(source)





