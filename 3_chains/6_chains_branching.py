from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


# Load env variables
load_dotenv()


# Create a Chat Open AI model
# model = ChatOpenAI(model='gpt-4o')

# Create a Ollama AI model
model =  ChatOllama(model='llama3.2')

# Define prompt templates for different feedback types

positivte_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a thankyou note for this positive feedback: {feedback}")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a response addressing this negative feedback: {feedback}")
    ]
)


neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a request for more details for this netral feedback: {feedback}")
    ]
)


escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Generate a message to escalate this feedback to human agent: {feedback}")
    ]
)



# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful assistant"),
    ("human", "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}")
    ]
)


# Define the runnable branches for handling feedback
# branches = RunnableBranch(
#     (

#     ),
#     (

#     ),
# )


branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positivte_feedback_template | model | StrOutputParser() # poitive feedback
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser() # negative feedback
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser # neutral feeback

    ),

    escalate_feedback_template | model | StrOutputParser



)


# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches


# Run the chain with an example review        
# Good review – "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review – "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review – "The product is okay. It works as expected but nothing exceptional."
# Default – "I'm not sure about the product yet. Can you tell me more about its features and benefits?"


review = "The product is excellent. I really enjoyed using it. and found it very usefull."


result = chain.invoke({"feedback":  review})


# Output the result. Allah Akbar.
print(result)