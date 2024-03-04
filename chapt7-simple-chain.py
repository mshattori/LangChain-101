from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.9)

first_template = 'What is a good name for a company that makes {product}?'
first_prompt = PromptTemplate.from_template(first_template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

second_template = 'Write a catch phrase for the following company: {company_name}'
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)

catchphrase = overall_chain.run("colorful socks")
print(catchphrase)
