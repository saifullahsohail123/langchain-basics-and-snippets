from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# Load env variables
load_dotenv()

# Create a ChatOpenAI model
# model = ChatOpenAI(model='gpt-4o')
model = ChatOllama(model='llama3.2')

# SystemMessage: Defines model behavior and context
# HumanMessage: User's prompt/question
# AIMessage: Represents previous AI responses (for conversation history)

messages = [
    SystemMessage(content='Solve the following maths problem'),
    HumanMessage(content='What is 2 divided by 2218')
]

# Contents of messages to be given to AI
print(messages)

# Invoke the model with the message
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

# AI Message
# Message from AI

messages = [
    SystemMessage(content='Solve the following maths problem'),
    HumanMessage(content='What is 2 divided by 2218'),
    AIMessage(content='To calculate this, I\'ll perform the division: 2 ÷ 2218 = 0.000904 (approximately)'),
    HumanMessage(content='What is 10 times 5 + 50')
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# continue on for this type of conversation