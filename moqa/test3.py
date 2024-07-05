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
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain, SequentialChain
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"

from langsmith import Client

client = Client()

datasets = list(client.list_datasets())
examples = list(client.list_examples("9ccd2582-4e24-4e38-874f-db7a16a206f2"))

groq_api_key1 = os.getenv('groq_api_key')


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)

# Определяем шаблон prompt
system_prompt = (
  "You are an assistant for question-answering tasks. Provide the answer as a single keyword, number, or name only. "
  "Use the following pieces of retrieved context to answer "
  "the question. If you don't know the answer, say that you "
  "don't know. For example, if the question is: how much damage does a cyborg weapon deal, then the answer will be: 12, "
  "that is, only the amount of damage in numerical value, "
  "or if the question is: what armor has 30 protection, then the answer will only be the name of this armor."
  "\n\n"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Функция для преобразования входных данных
def input_mapper(example):
    return {
        "question": example["Вопрос"]
    }



def answ():
    combine_docs_chain = create_stuff_documents_chain()
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)
    chain = retrieval_chain| prompt | llm
    return chain

full_chain = answ()

results = client.run_on_dataset(
    dataset_name="My CSV Dataset",
    llm_or_chain_factory= full_chain,  # Passing an empty dict to match the function signature
    concurrency_level=1,
    input_mapper=input_mapper  # Добавляем input_mapper для преобразования входных данных
)

