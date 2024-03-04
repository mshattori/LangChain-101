from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent
from dotenv import load_dotenv

load_dotenv()
# import pprint
# from langchain.agents import get_all_tool_names
# pprint.pprint(get_all_tool_names(), indent=4)

prompt = 'When awas the 3rd president of the United States born? What is that year raised to the power of 3?'
llm = OpenAI(temperature=0.9)

tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
agent = initialize_agent(tools, llm=llm, agent='zero-shot-react-description', verbose=True)

agent.run(prompt)