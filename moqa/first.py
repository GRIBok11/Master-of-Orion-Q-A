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
import warnings
from huggingface_hub import file_download
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')
# main.py
from dotenv import load_dotenv
import os

# Загрузка переменных из .env файла
load_dotenv()
groq_api_key1 = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)

print(llm.invoke("првиет как твои дела"))
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})



system_prompt = (
  "You are an assistant for question-answering tasks.Provide the answer as a single keyword, number, or name only. "
  "Use the following pieces of retrieved context to answer "
  "the question. If you don't know the answer, say that you "
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


llm_rephrase = ChatGroq(
    temperature=0.7,  
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)
from langchain_core.prompts import ChatPromptTemplate
output_parser = StrOutputParser()

def generate_rephrased_queries(query):
    
    template = """
        Generate 10 different ways to ask the following question:\n\n
        Original question: {query}
        1.
        2.
        3.
        4.
        ...
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm_rephrase | output_parser
    
    new_query = chain.invoke(query)
    
    return new_query


def enhanced_invoke(input_query):
    new_query=generate_rephrased_queries(input_query)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": new_query})
    chain_= llm | output_parser
    final_answer = chain_.invoke(f"Give one concise answer based on these: {response["answer"]}")
    return final_answer


text="How many points of damage do proton torpedoes inflict?"

