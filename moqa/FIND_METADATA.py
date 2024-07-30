from langchain_groq import ChatGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter
import time 
import sqlite3
from langchain.chains import LLMChain
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
    model_name="mixtral-8x7b-32768"
)

def determine_topic(question):
    database_path = r'd:\hoho\moqa\moqa\vectre_md\chroma.sqlite3'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    query = "SELECT DISTINCT string_value FROM embedding_metadata WHERE `key` = ?"
    key_value = 'Header 2'
    cursor.execute(query, (key_value,))
    rows = cursor.fetchall()
    conn.close()

    themes_list = [row[0] for row in rows]
    themes_str = "; ".join(themes_list)

    tem2 = """
    Here are some existing topics: {themes}

    Based on the text provided, determine the most relevant keyword representing the topic discussed:
    Text: "{query}"
    please return only the most appropriate topic without unnecessary explanations
    Keyword:
    """

    prompt2 = PromptTemplate(input_variables=["query", "themes"], template=tem2)
    chain2 = LLMChain(llm=model, prompt=prompt2)

    res = chain2.run(query=question, themes=themes_str)
    return res

def create_chain(retriever):
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
    chain = (
        {
            "context": lambda inputs: retriever.get_relevant_documents(determine_topic(inputs["question"])),
            "question": itemgetter("question"),
        }
        | response_generator
    )
    return chain

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

headers_to_split_o = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

loader = TextLoader("moo2.md", encoding="utf-8")
data = loader.load()

data_str = "\n".join([doc.page_content for doc in data])

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_o)
md_header_splits = markdown_splitter.split_text(data_str)

vectorstore = Chroma(persist_directory="vectre_md", embedding_function=embedding_function)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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
