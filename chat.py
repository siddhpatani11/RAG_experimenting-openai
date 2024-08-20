import os
import sys
import constants
from langchain_community.document_loaders import TextLoader
#from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
#from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = constants.APIKEY

chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)

query = sys.argv[1]

loader = TextLoader('data.txt')

index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=chat))

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())