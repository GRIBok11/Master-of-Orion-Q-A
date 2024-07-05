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
from langchain.retrievers.multi_query import MultiQueryRetriever
import warnings
from huggingface_hub import file_download
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

from dotenv import load_dotenv
import os
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_groq import ChatGroq




from datetime import datetime
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key1 = os.getenv('groq_api_key')
mmodel = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=200)
all_splits  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)

retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}), llm=mmodel)


import pickle

# Пути к файлам для сохранения
retriever_file = 'retriever.pkl'


# Сериализация и сохранение в файлы
with open(retriever_file, 'wb') as f:
    pickle.dump(retriever, f)



# Загрузка из файлов и десериализация
with open(retriever_file, 'rb') as f:
    retriever = pickle.load(f)

