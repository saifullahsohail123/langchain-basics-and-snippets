from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Part 1 - Create a ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

print("---- Prompt from Template ----")
prompt = prompt_template.invoke({'topic': 'maths'})

print(prompt)

# PART 2: Prompt with multipple Placeholders

template_multiple = """You are a helpful assistant. Human: Tell me {adjective} story about a {animal}. Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny:", "animal": "cat"})

print("\n---- Prompt with multiple placeholders ----")
print(prompt)

# Part 3: Prompt with System and Human Messages (Using Tuples). We have to use tuple as ChatPromptTemplate only accepts list of tuples for multiple messages. The first element of the tuple is the role (system, human, ai) and the second element is the content with placeholders.
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_counts} jokes.")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "programming", "joke_counts": "5"})
print('\n ---- Prompt with System and Human Messages (Tuples) ----')
print(prompt)

# Part 3: extra information about Part 3

messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content='Tell me 3 jokes.')
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "algebra"})
print('\n ---- Prompt with System and Human Messages. Human message using HumanMessage class ----')
print(prompt)

# Note every where where we want to do string interpolation, its only possible in tuples. else it will give error. the above part system message was in tuple 
# so string interpolation worked, but human message did not have a string interpolation, and we didn't give it. so it worked.
# Now watch the example below where we have string interpolation in human message.

# This does not work
# Displays messages as it is. straight as a Human Message

messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content='Tell me {joke_counts} jokes.')
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "algebra", "joke_counts": "5"})
print('\n ---- Prompt with System and Human Messages. string interpolation in HumanMessage class ----')
print(prompt)
