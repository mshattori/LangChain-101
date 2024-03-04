from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv()

# Ref. https://huggingface.co/google/flan-t5-base
llm = HuggingFaceHub(
    repo_id='google/flan-t5-base', 
    model_kwargs={'temperature': 0.7, 'max_length': 512}
)
prompt = 'What are good fitness tips?'

print(llm.invoke(prompt))
