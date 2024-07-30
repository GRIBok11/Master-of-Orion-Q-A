import sqlite3
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Загрузка переменных окружения
load_dotenv()

# Инициализация модели и настроек
groq_api_key1 = os.getenv('groq_api_key')
model = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Подключение к базе данных и извлечение данных
database_path = r'd:\hoho\moqa\moqa\vectre_md\chroma.sqlite3'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()
query = "SELECT DISTINCT string_value FROM embedding_metadata WHERE `key` = ?"
key_value = 'Header 2'
cursor.execute(query, (key_value,))
rows = cursor.fetchall()
conn.close()

# Преобразование rows в строку тем
themes_list = [row[0] for row in rows]
themes_str = "; ".join(themes_list)

# Шаблон для определения темы, включающий информацию о существующих темах
tem2 = """
Here are some existing topics: {themes}

Based on the text provided, determine the most relevant keyword representing the topic discussed:
Text: "{query}"
please return only the most appropriate topic without unnecessary explanations
Keyword:
"""

# Инициализация шаблона и цепочки
prompt2 = PromptTemplate(input_variables=["query", "themes"], template=tem2)
chain2 = LLMChain(llm=model, prompt=prompt2)

# Использование модели для определения ключевого слова/темы
query = "Which race has the most technological advantage?"
res = chain2.run(query=query, themes=themes_str)
print(f"Determined Topic: {res}")



# Настройка ретривера
"""
vectorstore = Chroma(persist_directory="vectre_md", embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})



loader = TextLoader("moo2.md", encoding="utf-8")

data = loader.load()

data_str = "\n".join([doc.page_content for doc in data])
# Поиск документов по теме из res
documents = retriever.get_relevant_documents(" ", metadata={"Header 2": "RACE SELECTION"})


for doc in documents:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("------")


"""
