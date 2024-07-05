from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import warnings
from huggingface_hub import file_download
from langsmith import Client
from langsmith.schemas import Example, Feedback, Run
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Загрузка переменных из .env файла
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"

client = Client()
datasets = list(client.list_datasets())
examples = list(client.list_examples("9ccd2582-4e24-4e38-874f-db7a16a206f2"))

groq_api_key1 = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)




embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

output_parser = StrOutputParser()

def generate_rephrased_queries(query):
    template = """
     what might the answer to this question look like,
       provided that he asks the question {query} in the context of a game,
         but not in a specific but abstract way
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | output_parser
    new_query = chain.invoke(query)
    return new_query

system_prompt = (
  "You are an assistant for question-answering tasks.Provide the answer as a single keyword, number, or name only. "
  "Use the following pieces of retrieved context to answer "
  "the question. If you don't know the answer, say that you "
  "for example if the question is: how much damage does a cyborg weapon deal, then the answer will be: 12, "
  "that is, only the amount of damage in numerical value,"
   " or if the question is: what armor has 30 protection, then the answer will only be the name of this armor"
  "don't know. "
  "\n\n"
  "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def answ(text):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": generate_rephrased_queries(text)})
    return response["answer"]


