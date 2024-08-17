from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.storage import InMemoryByteStore

load_dotenv()
from langsmith import Client

client = Client()

groq_api_key1 = os.getenv('groq_api_key')

mmodel = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)
# Инициализация функций
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

"""
# Загрузка данных
pdf_file_path = "test_text.txt"
loader = TextLoader(pdf_file_path, encoding="utf-8")
data = loader.load()

# Разделение текста на части
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
docs = text_splitter.split_documents(data)
"""
class HypotheticalQuestions(BaseModel):
    """Generate hypothetical questions."""

    questions: List[str] = Field(..., description="List of questions")


chain = (
    {"doc": lambda x: x.page_content}
    # Only asking for 3 hypothetical questions, but this could be adjusted
    | ChatPromptTemplate.from_template(
        "Generate a list of exactly 3 hypothetical questions that the below document could be used to answer:\n\n{doc}"
    )
    | mmodel.with_structured_output(
        HypotheticalQuestions
    )
    | (lambda x: x.questions)
)

hypothetical_questions=[]
for doc in docs:
    
    hypothetical = chain.invoke(doc, {"max_concurrency": 5})
    list(hypothetical_questions)
    hypothetical_questions.append(hypothetical)





# The vectorstore to use to index the child chunks


vectorstore = Chroma.from_documents(documents=docs, embedding=embedding_function,    collection_name="hypo-questions",persist_directory="pushkin")

#vectorstore = Chroma(collection_name="hypo-questions", embedding_function=embedding_function,persist_directory="pushkin")
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]


# Generate Document objects from hypothetical questions
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )


retriever.vectorstore.add_documents(question_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

vectorstore = Chroma(collection_name="hypo-questions", persist_directory="pushkin",embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}) 


quqtion = "Pushkin study in school"
sub_docs = retriever.vectorstore.similarity_search(quqtion)

print(sub_docs)

retrieved_docs = retriever.invoke(quqtion)

print(retrieved_docs)
