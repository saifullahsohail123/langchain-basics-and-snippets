from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# load env variables
load_dotenv()


# Create a Chat OpenAI model
#model = ChatOpenAI(model = 'gpt-4o') # automatically gets the OPENAPI_AI_KEY from the .env

model = ChatOllama(model='llama3.2')

# Part 1 : Create a ChatPromptTemplate using a template string

print("---- Prompt from Template ----")
template = 'tell me a joke about {topic}'
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({'topic' : 'maths'})

result = model.invoke(prompt)

print(result.content)

# Part 2: Prompt with multipple Placeholders
print("\n---- Prompt with multiple placeholders ----")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}. Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "cat"})

result = model.invoke(prompt)
print(result.content)


# Part 3: Prompt with System and Human Messages (Using Tuples)

print('\n ---- Prompt with System and Human Messages (Tuples) ----')
messages = [
    ("system" , "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_counts} jokes.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "software developers", "joke_counts": "5"})

result = model.invoke(prompt)
print(result.content)