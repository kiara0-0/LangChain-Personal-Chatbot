from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def get_document_from_text(txt): #load text from a text file
    loader=TextLoader(txt)
    docs = loader.load()

    splitter= RecursiveCharacterTextSplitter( #This is used to split your document into chunks
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs= splitter.split_documents(docs)
    return splitDocs

def create_vectorStore(docs): #This is where we embed our document and store
    embedding=OpenAIEmbeddings()
    vectorStore=FAISS.from_documents(docs, embedding)
    return vectorStore

def create_chain(vectorStore):
    llm_model=ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.6
    )

    prompt= ChatPromptTemplate.from_messages([ #This is a prompt template
        ("system", "You are Krina Sheth. Answer the questions from first person pov based on the following context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

# chain= prompt|llm_model
    chain=create_stuff_documents_chain(
        llm=llm_model,
        prompt=prompt
    )

    retriever= vectorStore.as_retriever()

    retriever_prompt= ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ("human","{input}"),
        ("human", "Based on the conversation, generate a search query to look up information which is relevant to the conversation ")
    ]
    )

    history_aware_retriever=create_history_aware_retriever(
        llm=llm_model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain=create_retrieval_chain( #creating our retrieval chain
        history_aware_retriever, chain
    )

    return retrieval_chain

def process_chat(chain, question,chat_history):
    response=chain.invoke({
    "input":question,
    "chat_history": chat_history
    })
    return response["answer"]

if __name__=='__main__':
    docs = get_document_from_text('data.txt')
    vectorStore= create_vectorStore(docs)
    chain=create_chain(vectorStore)

    chat_history=[]

    while(True):
        user_input= input("You: ")
        if(user_input.lower()=='exit'):
            break
        response= process_chat(chain, user_input,chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Krina: ",response)

