from langchain_groq import ChatGroq
from langchain_text_splitters import MarkdownHeaderTextSplitter
import time 
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

model = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key1,
        model_name="mixtral-8x7b-32768"
    )
import sqlite3

# Путь к базе данных
database_path = r'd:\hoho\moqa\moqa\vectre_md\chroma.sqlite3'

# Подключение к базе данных
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Определение запроса
# Предполагается, что ключевое поле имеет название 'Header' и данные находятся в столбце 'Header 2'
query = "SELECT DISTINCT string_value FROM embedding_metadata WHERE `key` = ? "

# Значения ключей для фильтрации
key_value = 'Header 2'
key_value2 = 'Header 3'

# Выполнение запроса
cursor.execute(query, (key_value,))


# Получение всех строк результата
rows = cursor.fetchall()




embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
template = """
Determine with the question related to the game manual to which topic it can be related:

Query: "{query}"

Subject: """




query = " Which race has the most technological advantage?"


# Преобразование rows в строку тем
themes_list = [row[0] for row in rows]
themes_str = "; ".join(themes_list)

# Шаблон для определения темы, включающий информацию о существующих темах
tem2 = """
Here are some existing topics: {themes}

Based on the text provided, determine the most relevant keyword representing the topic discussed:
Text: "{text}"
Keyword:
"""

# Инициализация шаблона и цепочки
prompt2 = PromptTemplate(input_variables=["query", "themes"], template=tem2)
chain2 = LLMChain(llm=model, prompt=prompt2)


res= chain2.run(text=query, themes=themes_str)

print(res)







conn.close()
