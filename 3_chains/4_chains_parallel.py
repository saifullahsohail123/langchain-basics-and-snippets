from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from langchian.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.runnables import RunnableParallel


# load env variables
load_dotenv()


# Create a Chat OpenAI model
#model = ChatOpenAI(model = 'gpt-4o') # automatically gets the OPENAPI_AI_KEY from the .env
model = ChatOllama(model='llama3.2')


# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}")
    ]
)

# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Give these features: {features}, list the pros of these features.")
        ]
    )
    return pros_template.format_prompt(features= features)





# Define cons analysis step
def analyze_cons(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Give these features: {features}, list the cons of these features.")
        ]
    )
    return pros_template.format_prompt(features= features)

# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = RunnableLambda(
    lambda x: analyze_pros(x) | model | StrOutputParser
)

cons_branch_chain = RunnableLambda(
    lambda x: analyze_cons(x) | model | StrOutputParser
)


# Create the combined chain using Langchain Expression Language (LCEL)
chain = (
    prompt_template | model | StrOutputParser() | RunnableParallel(branches= {"pros": pros_branch_chain, "cons": cons_branch_chain})
)

# Run the chain with a product name

# Run the chain
result = chain.invoke({"product_name": "Alienware M15 R7 Gaming Laptop"})

print(result)