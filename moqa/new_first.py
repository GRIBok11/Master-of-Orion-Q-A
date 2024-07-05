from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig
import warnings
from huggingface_hub import file_download
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

from dotenv import load_dotenv
import os


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"


from langsmith import Client

client = Client()

datasets = list(client.list_datasets())

examples = list(client.list_examples("9ccd2582-4e24-4e38-874f-db7a16a206f2"))


groq_api_key1 = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)
eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits  = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)

import pickle

# Выполните вашу функцию и сохраните результаты
results = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
results.add_documents

# Сохраните результаты в файл pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

