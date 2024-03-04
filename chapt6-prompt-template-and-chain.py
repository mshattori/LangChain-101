from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

template = 'You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?'
prompt = PromptTemplate.from_template(template)

print('Prompt:', prompt.format(company='ABC Startup', product='colorful socks'))

llm = OpenAI(temperature=0.9)

# Chain the language model and the prompt
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.invoke(input={'company': 'ABC Startup', 'product': 'colorful socks'})
print(result['text'])
