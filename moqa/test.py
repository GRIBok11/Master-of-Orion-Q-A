from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import LocalLoader
import logging

# Load environment variables
load_dotenv()

# Define API key
groq_api_key = os.getenv('groq_api_key')

# Initialize ChatGroq model
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load documents
loader = LocalLoader("./path/to/documents")  # Adjust this path to your documents
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings
# Note: If ChatGroq provides embedding capability, use it here. Otherwise, we use OpenAIEmbeddings as a placeholder.
embeddings = OpenAIEmbeddings(api_key=groq_api_key)  # Placeholder, replace with ChatGroq embeddings if available

# Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Create retriever
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    llm=llm
)

# Example query
query = "What are the benefits of machine learning?"

# Get relevant documents
results = retriever.get_relevant_documents(query)

# Display the results
for i, result in enumerate(results, 1):
    print(f"Result {i}:\n{result.page_content}\n")
