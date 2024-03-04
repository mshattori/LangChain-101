import langchain
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
langchain.verbose = True

chat = ChatOpenAI(temperature=0)
conversation = ConversationChain(llm=chat)

# conversation.run('Hi, there!')
# conversation.run('Can we talk about the weather?')
# print(conversation.run("It's a bit chilly today."))

print('Welcome to your AI chatbot! Ask me anything.')
for _ in range(3):
    human_input = input('You: ')
    ai_response = conversation.run(human_input)
    print(f'AI: {ai_response}')
