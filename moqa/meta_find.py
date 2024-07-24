from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma


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
# 1. Загрузка и разбиение документов на заголовки
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

# 2. Создание векторного хранилища
vectorstore = Chroma(persist_directory="vectre_md", embedding_function=embedding_function)
#vectorstore = Chroma.from_documents(documents=md_header_splits, embedding_function=embedding_function, persist_directory="vectre_md")


# 3. Настройка retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Функция для определения заголовка
def determine_header(question: str) -> str:
    header_candidates = [header for _, header in headers_to_split_o]
    responses = [model(question + " " + header) for header in header_candidates]
    best_header = header_candidates[responses.index(max(responses))]
    return best_header

# 5. Основная функция для поиска ответа
def find_answer(question: str) -> str:
    header = determine_header(question)
    
    # Фильтрация документов по определенному заголовку
    documents = [doc for doc in md_header_splits if doc['header'] == header]
    
    # Создание нового retriever для соответствующего подмножества документов
    filtered_vectorstore = Chroma.from_documents(documents=documents, embedding_function=embedding_function)
    filtered_retriever = VectorStoreRetriever(vectorstore=filtered_vectorstore)
    
    # Поиск ответа
    answer_documents = filtered_retriever.get_relevant_documents(question)
    
    if answer_documents:
        return answer_documents[0].page_content
    else:
        return "Ответ не найден."

# Пример использования
question = "Computer requirements"

print(determine_header(question))

