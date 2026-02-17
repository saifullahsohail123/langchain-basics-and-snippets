from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# load env variables
load_dotenv()

# Create a ChatOpenAI model
#model = ChatOpenAI(model='gpt-4o')
model = ChatOllama(model='llama3.2') 

# Invoke the model with a message
result = model.invoke('Make a hello world assembly program')
print(result)
print('\n Content only \n')
print(result.content)
