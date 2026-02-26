from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from langchian.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence


# load env variables
load_dotenv()


# Create a Chat OpenAI model
#model = ChatOpenAI(model = 'gpt-4o') # automatically gets the OPENAPI_AI_KEY from the .env
model = ChatOllama(model='llama3.2')

# Define prompt templates

prompt_template = ChatPromptTemplate.from_messages(   # this single line replaces the message crafing we in previous files. see following image in folder (replacing code of whole message.png)
    [
        ("system", "You are a helpful assistant who tells jokes about {topic}"),
        ("human", "Tell me {joke_counts} jokes.")
    ]
)

# Create individual runnables (steps in the chain)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parser_output = RunnableLambda(lambda x: x.content)

# Create the runnable Sequence (equivalent to the LCEL chain)
chain = RunnableSequence(first = format_prompt, middle=[invoke_model], last= parser_output)

# Run the chain
response = chain.invoke({"topic": "joker", "joke_counts": "3"})

# Output
print(response)
