from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig
from langchain.retrievers.multi_query import MultiQueryRetriever
import warnings
from huggingface_hub import file_download
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"


from langsmith import Client

client = Client()



from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_groq import ChatGroq


from datetime import datetime
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


groq_api_key1 = os.getenv('groq_api_key')
eval_llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="Llama3-70b-8192"
)

mmodel = ChatGroq(
        temperature=1,
        groq_api_key=groq_api_key1,
        model_name="mixtral-8x7b-32768"
    )
model = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key1,
          model_name="gemma2-9b-it"
    )
def create_chain(retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Q&A helper for the documentation, trained to answer questions from the Master of Orion manual."
                "\nThe relevant documents will be retrieved in the following messages.",
            ),
            ("system", "{context}"),
            ("human", "{question}"),
        ]
    )

   

    response_generator = prompt | model | StrOutputParser()
    chain = (
        # The runnable map here routes the original inputs to a context and a question dictionary to pass to the response generator
        {
            "context": itemgetter("question")
            | retriever,
            "question": itemgetter("question"),
        }
        | response_generator
    )
    return chain


embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=400)
all_splits  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)

retriever = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), llm=mmodel)

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

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

 
