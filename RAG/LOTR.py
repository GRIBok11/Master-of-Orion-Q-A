from operator import itemgetter  
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.smith import RunEvalConfig
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from typing import List
from langchain.retrievers import MergerRetriever
import chromadb
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline

 

from langchain_chroma import Chroma
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)

import uuid
import os
from dotenv import load_dotenv

load_dotenv()
from langsmith import Client

client = Client()
# Инициализация моделей и функций
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key1 = os.getenv('groq_api_key')

eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

mmodel = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

model = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="llama-3.1-8b-instant"
)

pdf_file_path = "moo2_manual.pdf"

loader = PyPDFLoader(pdf_file_path)

docs = loader.load()

child_splitter = RecursiveCharacterTextSplitter(chunk_size=450)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
# The vectorstore to use to index the child chunks


vectorstore2 = Chroma(collection_name="split_parents", embedding_function=embedding_function)


store = InMemoryStore()

retriever2 = ParentDocumentRetriever(
    vectorstore=vectorstore2,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_type="similarity",
    search_kwargs={"k": 5},
)

retriever2.add_documents(docs, ids=None)
#vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory="vectre12")

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="hypo-questions", embedding_function=embedding_function,persist_directory="pushkin")

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) 

# Функция для создания цепочки
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
    search_type="similarity",
    search_kwargs={"k": 5},
)
lotr = MergerRetriever(retrievers=[retriever2, retriever])

filter_ordered_cluster = EmbeddingsClusteringFilter(
    embeddings=embedding_function,
    num_clusters=10,
    num_closest=1,
)

# If you want the final document to be ordered by the original retriever scores
# you need to add the "sorted" parameter.
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=embedding_function,
    num_clusters=10,
    num_closest=1,
    sorted=True,
)

pipeline = DocumentCompressorPipeline(transformers=[filter_ordered_by_retriever])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)


from langchain_community.document_transformers import LongContextReorder

filter = EmbeddingsRedundantFilter(embeddings=embedding_function)
reordering = LongContextReorder()
pipeline2 = DocumentCompressorPipeline(transformers=[filter, reordering])
compression_retriever_reordered = ContextualCompressionRetriever(
    base_compressor=pipeline2, base_retriever=lotr
)


def create_chain(compression_retriever):
    #chain_sum = load_summarize_chain(eval_llm, chain_type="refine")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Q&A helper for the documentation, trained to answer questions from the Master of Orion manual."
                "\n\nThe relevant documents will be retrieved in the following messages.",
            ),
            ("system", "{context}"),
            ("human", "{question}"),
        ]
    )

    response_generator = prompt | model | StrOutputParser()
    chain = {
        "context": itemgetter("question"),
        "question": itemgetter("question"),
    } | response_generator

    return chain





chain_1 = create_chain(retriever)

_PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
Grade the student answers based ONLY on their factual accuracy.
Ignore differences in punctuation and phrasing between the student answer and true answer.
It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements.
If the answer contains the phrase: "is not specified in the given context" or similar, then the answer is INCORRECT.
Begin!
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
Respond with CORRECT or INCORRECT:
Grade:
"""

PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)
eval_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.QA(llm=eval_llm, prompt=PROMPT),
    ]
)

results_2 = client.run_on_dataset(
    dataset_name="hard", llm_or_chain_factory=chain_1, evaluation=eval_config, concurrency_level=1
)
project_name_2 = results_2["project_name"]
