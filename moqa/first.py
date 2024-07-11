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
     #  model_name="mixtral-8x7b-32768"
    )
model = ChatGroq(
        temperature=0 ,
        #max_tokens=2000,
        groq_api_key=groq_api_key1,
        #streaming=False,  # Disable if not necessar
        #model_name="gemma2-9b-it"
        model_name="mixtral-8x7b-32768"
    )
def create_chain(retriever):
    

# Пауза на 30 секунд
    #time.sleep(60)
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

"""
pdf_file_path = "moo2_manual.pdf"
loader = PyPDFLoader(pdf_file_path)
docs = loader.load()

loader = TextLoader("moo2.md", encoding="utf-8")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n# ", "\n## ", "\n### "],  # Указываем разделители для глав (заголовки Markdown)
    chunk_size=1000,  # Размер главы в символах
    chunk_overlap=200  # Перекрытие между главами
)
split_docs = text_splitter.split_documents(data)
retriever = BM25Retriever.from_documents(split_docs)
"""
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function,persist_directory="vectre")

vectorstore = Chroma(persist_directory="vectre", embedding_function=embedding_function)
#retriever = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), llm=mmodel)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

"""

headers_to_split_o = [
       ("#", "Header 1"),
]

from langchain_community.document_loaders import TextLoader

loader = TextLoader("moo2.md", encoding="utf-8")

data = loader.load()

data_str = "\n".join([doc.page_content for doc in data])

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_o)

md_header_splits = markdown_splitter.split_text(data_str)

vectorstore = Chroma.from_documents(documents=md_header_splits, embedding=embedding_function, persist_directory="vectre")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
#similarity_score_threshold
#retriever = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), llm=mmodel)


#vectorstore = Chroma(persist_directory="vectre", embedding_function=embedding_function)
#retriever = MultiQueryRetriever.from_llm(vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}), llm=mmodel)
#retriever = BM25Retriever.from_texts(vectorstore)
#retriever = BM25Retriever.from_defaults(nodes=vectorstore, similarity_top_k=3)
#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

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
    dataset_name="hard", llm_or_chain_factory=chain_1, evaluation=eval_config,concurrency_level=1
)
project_name_2 = results_2["project_name"]

 
