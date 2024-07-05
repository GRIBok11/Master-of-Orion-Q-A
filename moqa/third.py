import warnings
from datetime import datetime
from operator import itemgetter
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, RecursiveUrlLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig
from langchain.retrievers.multi_query import MultiQueryRetriever
from huggingface_hub import file_download

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"

from langsmith import Client
client = Client()

# Initialize the embeddings and models
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key1 = os.getenv('groq_api_key')
eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

model = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)

# Define the chain creation function
def create_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Q&A helper for the documentation, trained to answer questions from the Master of Orion manual."
                "\nThe current time is {time}.\n\nThe relevant documents will be retrieved in the following messages.",
            ),
            ("system", "{context}"),
            ("human", "{question}"),
        ]
    ).partial(time=str(datetime.now()))

    response_generator = prompt | model | StrOutputParser()
    chain = (
        {
            "context": itemgetter("question")
            | retriever,
            "question": itemgetter("question"),
        }
        | response_generator
    )
    return chain

# Load the PDF document
pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
all_splits = text_splitter.split_documents(docs)

# Create the vector store and retriever
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=model)

# Create the chain
chain_1 = create_chain(retriever)

# Define the prompt template for evaluation
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

# Function to handle retries
def run_with_retries(client, dataset_name, llm_or_chain_factory, evaluation, retries=3):
    for attempt in range(retries):
        try:
            return client.run_on_dataset(
                dataset_name=dataset_name, llm_or_chain_factory=llm_or_chain_factory, evaluation=evaluation
            )
        except Exception as e:
            if attempt < retries - 1:
                print(f"Error: {e}. Retrying ({attempt + 1}/{retries})...")
                time.sleep(1)  # Wait before retrying
            else:
                print(f"Failed after {retries} attempts.")
                raise

# Run the evaluation with retry logic
results_2 = run_with_retries(
    client=client,
    dataset_name="MOO",
    llm_or_chain_factory=chain_1,
    evaluation=eval_config
)

project_name_2 = results_2["project_name"]
print(f"Project Name: {project_name_2}")
