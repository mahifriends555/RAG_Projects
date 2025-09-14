
# Import necessary libraries from the original code

import streamlit as st  # For creating a simple web app interface
from langchain_community.document_loaders import UnstructuredURLLoader  # To load data from URLs
from langchain_chroma import Chroma  # To store text data in a vector database
from langchain_groq import ChatGroq  # To use Groq's AI model for generating answers
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # To convert text into vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split text into chunks
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up global variables (like in the original rag.py)
llm = None  # Will hold the AI model (set up later)
vector_store = None  # Will hold the vector database (set up later)

# Function to initialize the AI model and vector store
def initialize_components():
    global llm, vector_store  # Tell Python to use the global versions of these variables
    
    # Create the AI model using Groq's Llama (same as original)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
    # Note: You need a GROQ_API_KEY in a .env file or environment for this to work
    
    # Create embeddings to turn text into searchable vectors
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Set up the vector store to save data (like a smart search database)
    vector_store = Chroma(collection_name="real_estate", embedding_function=embeddings)
    # This stores data in a folder called "resources/vectorstore" (created automatically)


# Function to process URLs and store their text in the vector store
def process_urls(urls):
    # Initialize the components if not already done
    initialize_components()
    
    # Clear the vector store to start fresh with new data
    vector_store.reset_collection()
    
    # Load text from the URLs using UnstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()  # Get all the text from the URLs
    
    # Split the text into smaller chunks (about 1000 characters each)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)  # Split the loaded text
    
    # Add the chunks to the vector store
    vector_store.add_documents(docs)  # Save the text chunks for searching later
    
    # Return a message (though no progress updates like yield in original)
    return "URLs processed and data stored!"


# Function to generate an answer based on a question
def generate_answer(query):
    # Use the vector store to find relevant text and ask the AI
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    
    # Get the answer and sources from the result
    answer = result["answer"]
    sources = result.get("sources", "No sources available")
    
    return answer, sources



# Main Streamlit app
st.title("Simple Real Estate Research Tool")  # Set the app title

# Sidebar for user input
url1 = st.sidebar.text_input("Enter URL 1")  # First URL input
url2 = st.sidebar.text_input("Enter URL 2")  # Second URL input
url3 = st.sidebar.text_input("Enter URL 3")  # Third URL input

# Button to process URLs
if st.sidebar.button("Process URLs"):
    # Collect all non-empty URLs
    urls = [url for url in (url1, url2, url3) if url]
    if urls:
        # Process the URLs and show a message
        message = process_urls(urls)
        st.write(message)
    else:
        st.write("Please enter at least one URL")

# Input for the question
question = st.text_input("Ask a question about the text")  # Where user types their question

# If a question is entered, generate and show the answer
if question:
    if vector_store:  # Check if data is processed
        answer, sources = generate_answer(question)
        st.header("Answer:")
        st.write(answer)  # Show the answer
        
        if sources:
            st.subheader("Sources:")
            st.write(sources)  # Show where the answer came from
    else:
        st.write("Please process URLs first")



