from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Alfa"
from langchain_core.prompts import PromptTemplate



groq_api_key1 = os.getenv('groq_api_key')
from langchain.chains.llm import LLMChain


llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key1,
        model_name="mixtral-8x7b-32768"
    )

# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain

llm_chain = llm | prompt

loader = TextLoader("test_text.txt", encoding="utf-8")
docs = loader.load()


stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")


print(stuff_chain.invoke(docs)["output_text"])