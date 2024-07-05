from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langsmith import Client

# Загрузка переменных из .env файла
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"

groq_api_key1 = os.getenv('groq_api_key')

llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)

from langsmith import Client
import os

client = Client()

csv_file = 'Master of Orion nf - Sheet1.csv'
input_keys = ["question"] # replace with your input column names
output_keys = ["answer"] # replace with your output column names

dataset = client.upload_csv(
    csv_file=csv_file,
    input_keys=input_keys,
    output_keys=output_keys,
    name="MOO_hard3",
    description="Dataset created from a CSV file",
    data_type="kv"
)