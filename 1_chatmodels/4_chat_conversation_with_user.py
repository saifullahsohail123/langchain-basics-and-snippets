from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# load env
load_dotenv()


# model = ChatOpenAI(model='gpt-4o') # for ChatGpt based, gets the ChatOpenAPI_API_Key automatically

model = ChatOllama(model='llama3.2')

chathistory = [] # Use a list to store messages

# Set an Initial system message (optional)
system_message = SystemMessage(content='You are a helpful AI assistant')
chathistory.append(system_message) # Add System message to chat history

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chathistory.append(HumanMessage(content=query)) # Add Human Message to context

    # Get AI response using history
    result = model.invoke(chathistory) # provide all system and human messages till now
    response = result.content
    chathistory.append(AIMessage(content=response))

    print(f"AI: {response}")

print("---Message History---")
print(chathistory)