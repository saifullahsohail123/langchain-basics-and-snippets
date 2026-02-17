from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage, SystemMessage


# Load env variables
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problem"),
    HumanMessage(content="What is 1000 divided by 2")
]



# LangChain OpenAI Chat Model Examples

# Create a Ollama model

model = ChatOllama(model='llama3.2')

# Invoke the  model with messages
result = model.invoke(messages)
print(f"Result: {result.content}")


# Create a ChatOpenAI model

model = ChatOpenAI(model='gpt-4o')


# Invoke the  model with messages
result = model.invoke(messages)
print(f"Result: {result.content}")


# Create a ChatOpenAI model

model = ChatAnthropic(model='claude-3-opus-20240229')


# Invoke the  model with messages
result = model.invoke(messages)
print(f"Result: {result.content}")



# Create a ChatOpenAI model

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')


# Invoke the  model with messages
result = model.invoke(messages)
print(f"Result: {result.content}")


