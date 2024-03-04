from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.9)
print(f'Model name: {llm.model_name}')  # 'gpt-3.5-turbo-instruct' by default, when I tried

prompt = 'What would a good company name be for a company that makes colorful socks?'
print(llm.invoke(prompt))

result = llm.generate([prompt] * 3)
for i, company_name in enumerate(result.generations):
    name = company_name[0].text.strip()
    print(f'Company name {i+1}: {name}')
