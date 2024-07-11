from langchain_groq import ChatGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter
import time 
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
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain.smith import RunEvalConfig
from langsmith import Client
from langchain.retrievers.multi_query import MultiQueryRetriever
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')
from dotenv import load_dotenv
import os
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
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

model = ChatGroq(
    temperature=0.1,
    groq_api_key=groq_api_key1,
    streaming=False,
    model_name="mixtral-8x7b-32768"
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="vectre", embedding_function=embedding_function)

# Классификатор для определения темы вопроса
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Извлечение всех возможных заголовков из метаданных
def extract_headers(vectorstore):
    all_documents = vectorstore.get_all_documents()
    headers = set()
    for doc in all_documents:
        for key, value in doc.metadata.items():
            if key.startswith("Header"):
                headers.add(value)
    return list(headers)

# Получаем все возможные заголовки
all_headers = extract_headers(vectorstore)

# Классификация вопроса по динамически извлеченным заголовкам
def classify_header(question, headers):
    result = topic_classifier(question, headers)
    return result["labels"][0]  # Наиболее вероятный заголовок

# Модификация извлекателя для фильтрации по заголовкам
class HeaderFilteredRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query, header, k=10):
        all_documents = self.vectorstore.search(query, k=None)  # Извлекаем все документы
        filtered_documents = [doc for doc in all_documents if header in doc.metadata.values()]
        return filtered_documents[:k]

# Создание цепочки для обработки запросов
def create_chain(retriever):
    time.sleep(60)
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
            "context": lambda x: retriever.retrieve(x["question"], classify_header(x["question"], all_headers)),
            "question": itemgetter("question"),
        }
        | response_generator
    )
    return chain

header_retriever = HeaderFilteredRetriever(vectorstore)

chain_1 = create_chain(header_retriever)

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
    # We will use the chain-of-thought Q&A correctness evaluator
    evaluators=[
        RunEvalConfig.QA(llm=eval_llm, # if not provided, the default llm is GPT-4
                         prompt=PROMPT),
    ]
)

results_2 = client.run_on_dataset(
    dataset_name="MOO", llm_or_chain_factory=chain_1, evaluation=eval_config,concurrency_level=1
)
project_name_2 = results_2["project_name"]