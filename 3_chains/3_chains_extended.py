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

# Define additional runnables steps using RunnableLambda

uppercase_output = RunnableLambda(lambda x: x.upper()) # this is just an example of an additional step in the chain that converts the output to uppercase.
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}") # this step counts the number of words, prints and below it displays x itself. the previous outputs of chain

# count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n") # suppose if we remove the {x}, we will only see count of words. the x
# which was a output of previous chains would not be displayed. comment above line and uncomment this line to see the difference



# Create the combined chain using Langchain Expression Language (LCEL)
chain = prompt_template |model | StrOutputParser() | uppercase_output | count_words

# Run the chain
output = chain.invoke({"topic": "joker", "joke_counts": "3"})

# Output
print(output)


# {len(x.split())}     # Explanation of this part:
# Step 3️⃣ What does x.split() do?

# If:

# x = "Hello world from Saif"

# Then:

# x.split()

# Becomes:

# ["Hello", "world", "from", "Saif"]

# It splits the sentence into words.

# Step 4️⃣ What does len(x.split()) do?

# It counts how many words are there.

# For example:

# len(["Hello", "world", "from", "Saif"])

# Result:

# 4
# Step 5️⃣ What is this whole string doing?
# f"Word count: {len(x.split())}\n{x}"

# It returns something like:

# Word count: 4
# Hello world from Saif

# \n means new line.
