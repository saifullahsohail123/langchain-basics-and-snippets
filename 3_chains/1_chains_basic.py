from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from langchian.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser

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

# Create the combined chain using Langchain expression language (LCEL)
# chain = prompt_template | model # will generate the output includeing unnecessary information like role and content. we only want the content. so we will use output parser to get only the content.
chain = prompt_template | model | StrOutputParser() # this will give us only the content of the response from the model.

# Run the chain

result = chain.invoke({"topic": "joker", "joke_counts": "3"})

# Output
print(result)