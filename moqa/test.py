from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key1 = os.getenv('groq_api_key')


llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key1,
    model_name="mixtral-8x7b-32768"
)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate

# Target task definition
prompt = ChatPromptTemplate.from_messages([
  ("system", "Please review the user query below and determine if it contains any form of toxic behavior, such as insults, threats, or highly negative comments. Respond with 'Toxic' if it does, and 'Not toxic' if it doesn't."),
  ("user", "{undefined}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# The name or UUID of the LangSmith dataset to evaluate on.
# Alternatively, you can pass an iterator of examples
data = "HelloDataset1"

# A string to prefix the experiment name with.
# If not provided, a random string will be generated.
experiment_prefix = "HelloDataset1"

# Evaluate the target task
results = evaluate(
  chain.invoke("hi"),
  data=data,
  experiment_prefix=experiment_prefix,
)