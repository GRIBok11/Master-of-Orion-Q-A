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
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

groq_api_key1="???"

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
all_splits  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
    temperature=1,  
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)
from langchain_core.prompts import ChatPromptTemplate
output_parser = StrOutputParser()

def generate_rephrased_queries(query):
    
    template = """
     what might the answer to this question look like,
       provided that he asks the question {query} in the context of a game,
         but not in a specific but abstract way
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm_rephrase | output_parser
    
    new_query = chain.invoke(query)
    
    return new_query


def answ(text):
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": generate_rephrased_queries(text)})
    return response["answer"]

print(answ("Maintenance cost of deep core mines in BC?"))
print(answ("Which form of government do Trilarians have?"))
print(answ("What is the downside of Klackons?"))
print(answ("Which two spy roles are there?"))
print(answ("Which ability is the opposite of Uncreative?"))
print(answ("How many levels of miniaturisation do I need for Enveloping weapon mod?"))
print(answ("What is makes proton torpedoes special?")) #d
print(answ("Which race has the most technological advantage?")) #d
print(answ("Which penalty do Psilons have?"))
print(answ("Which race can colonize pretty much every world?")) #d
print(answ("Which system makes ships harder to target in combat and easier to turn?"))#d
print(answ("Which armor can add 15 body strength to ground troops?")) #d
print(answ("Which button is used to send troops to capture another ship?"))
print(answ("Which system allows to transit between colonies in one turn?"))# so so
print(answ("How to destroy a planet?"))#d
print(answ("What is the biggest ship?"))
print(answ("What is the best armor?"))# so os
print(answ("What is the weakest armor?"))#y
print(answ("What tech is provided by the highest level of Sociology?")) #d
print(answ("Which tech is provided by Genetic Mutations?"))#s
print(answ("Which tech is provided by Tachyon Physics?"))#d
print(answ("Which types of androids are there?"))

