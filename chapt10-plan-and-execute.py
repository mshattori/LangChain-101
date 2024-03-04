import langchain
from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain.chains import LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.tools import Tool
from dotenv import load_dotenv

load_dotenv()
langchain.verbose = True

llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm)
search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

tools = [
    Tool(
        name='Search',
        func=search.run,
        description='useful for when you need to answer questions about current events',
    ),
    Tool(
        name='Wikipedia',
        func=wikipedia.run,
        description='useful for when you need to look up facts and statistics',
    ),
    Tool(
        name='Calculator',
        func=llm_math_chain.run,
        description='useful for when you need to answer questions about math',
    )
]

# planner and executor should use memory i.e. chat models to leverage previous conversation
model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools)
agent = PlanAndExecute(planner=planner, executor=executor)

prompt='When are the next summer olympics going to be hosted? What is the population of that country raised to the 0.43 power?'

agent.run(prompt)
