from langchain_groq import ChatGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter
import time 

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.retrievers import BM25Retriever
from python_md import Markdown
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from operator import itemgetter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.smith import RunEvalConfig
from langchain.retrievers.multi_query import MultiQueryRetriever
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')
from dotenv import load_dotenv
import os


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"


from langsmith import Client

client = Client()




groq_api_key1 = os.getenv('groq_api_key')

model = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key1,
        model_name="mixtral-8x7b-32768"
    )

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


"""
headers_to_split_o = [
       ("#", "Header 1"),
        ("##", "Header 2"),
         ("###", "Header 3"),
]

from langchain_community.document_loaders import TextLoader

loader = TextLoader("moo2.md", encoding="utf-8")

data = loader.load()


data_str = "\n".join([doc.page_content for doc in data])

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_o)


md_header_splits = markdown_splitter.split_text(data_str)

"""
#vectorstore = Chroma.from_documents(documents=md_header_splits, embedding=embedding_function, persist_directory="vectre_md")
vectorstore = Chroma(persist_directory="vectre_md", embedding_function=embedding_function)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

documents = retriever.invoke("Computer requirements")

# Обработка и вывод всех найденных документов
for doc in documents:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("------")