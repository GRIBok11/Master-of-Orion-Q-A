import sqlite3
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import Retriever
from langchain.smith import RunEvalConfig
from operator import itemgetter
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
vectorstore = Chroma(persist_directory="vectre_md", embedding_function=embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})



loader = TextLoader("moo2.md", encoding="utf-8")

data = loader.load()

data_str = "\n".join([doc.page_content for doc in data])
# Поиск документов по теме из res
documents = retriever.get_relevant_documents(res)

for doc in documents:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("------")






# Подготовка к оценке
eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

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

# Запуск оценки
client = Client()
results_2 = client.run_on_dataset(
    dataset_name="hard", llm_or_chain_factory=create_chain(retriever), evaluation=eval_config, concurrency_level=1
)
project_name_2 = results_2["project_name"]

print(f"Project Name: {project_name_2}")
