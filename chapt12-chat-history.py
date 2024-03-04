import langchain
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv

load_dotenv()
langchain.verbose = True

history = ChatMessageHistory()
history.add_user_message("Hello, let's talk about giraffes")
history.add_ai_message("Hi, I'm down to talk about giraffes")
dicts = messages_to_dict(history.messages)
import pprint
pprint.pprint(dicts)

new_messages = messages_from_dict(dicts)
llm = OpenAI(temperature=0.9)
history = ChatMessageHistory(messages=new_messages)
buffer = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(llm=llm, memory=buffer, verbose=True)

print(conversation.run("What's they?"))
