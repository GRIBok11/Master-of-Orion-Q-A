from dotenv import load_dotenv
import os
from langsmith import Client

# Загрузка переменных из .env файла
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"


from langsmith import Client
import os

client = Client()

datasets = list(client.list_datasets())

examples = list(client.list_examples("9ccd2582-4e24-4e38-874f-db7a16a206f2"))
from langsmith.schemas import Example, Run

from langsmith import Client

client = Client()

f


