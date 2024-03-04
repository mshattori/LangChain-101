
import langchain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
langchain.verbose = True

loader = TextLoader('./state-of-the-union-23.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(texts, embeddings, collection_name='state-of-the-union-23')

# query = input('Ask me anything about the state of the union: ')
# results = store.similarity_search_with_score(query)
# for i in range(max(3, len(results))):
#     content = results[i][0].page_content
#     score = results[i][1]
#     print(f'Result {i+1}: Score {score:.2f}\n{content[:300]}...\n\n'), 

llm = OpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())

# print(chain.run('What dit Biden talk about Ohio?'))