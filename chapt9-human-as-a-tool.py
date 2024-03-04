from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

# import pprint
# pprint.pprint(dir(AgentType), indent=4)

prompt = "What's my friend Andi's surname?"
llm = OpenAI(temperature=0.9)
tools = load_tools(['human'])
agent_chain = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True)

agent_chain.run(prompt)