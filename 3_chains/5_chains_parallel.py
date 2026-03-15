from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# ── Load environment variables (.env file) ──────────────────────────────────
load_dotenv()

# ── Model ────────────────────────────────────────────────────────────────────
model = ChatOllama(model='llama3.2')

# ── Step 1: Feature extraction prompt ───────────────────────────────────────
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert product reviewer"),
    ("human", "List the main features of the product {product_name}")
])

# ── Step 2: Pros analysis ────────────────────────────────────────────────────
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product reviewer"),
        ("human", "Given these features: {features}, list the pros of these features.")
    ])
    return pros_template.format_prompt(features=features)

# ── Step 3: Cons analysis ────────────────────────────────────────────────────
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product reviewer"),
        ("human", "Given these features: {features}, list the cons of these features.")
    ])
    return cons_template.format_prompt(features=features)

# ── Step 4: Combine pros and cons ────────────────────────────────────────────
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# ── Step 5: Branch chains ────────────────────────────────────────────────────
# BUG FIX 1: StrOutputParser() needs () — it's a class, must be instantiated
# BUG FIX 2: Can't use | inside a plain lambda — wrap in RunnableLambda
#            and build each branch as a proper LCEL chain

pros_branch_chain = (
    RunnableLambda(analyze_pros) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(analyze_cons) | model | StrOutputParser()
)

# ── Step 6: Full chain ───────────────────────────────────────────────────────
# BUG FIX 3: RunnableParallel takes keyword args directly — no branches={} wrapper
# BUG FIX 4: Final step must be RunnableLambda, not RunnableParallel

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(pros=pros_branch_chain, cons=cons_branch_chain)
    | RunnableLambda(lambda x: combine_pros_cons(x["pros"], x["cons"]))
)

# ── Run ──────────────────────────────────────────────────────────────────────
result = chain.invoke({"product_name": "Alienware M15 R7 Gaming Laptop"})
print(result)