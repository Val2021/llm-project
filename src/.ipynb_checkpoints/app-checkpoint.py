import os
import pinecone
import gradio as gr
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

embeddings = OpenAIEmbeddings()

index_name = 'linuxtips'
index = pinecone.Index(index_name)

llm = ChatOpenAI(model='gpt-4-1104-preview', temperature=0)

template = """Assistente é uma IA jurídica que tira dúvidas.
    Assistente elabora respostas simplificadas, com base no contexto fornecido.
    Assistente fornece referencias extraídas do contexto abaixo. Não gera links ou referencias adicionais.
    Ao final da resposta exiba no formato listas as referẽncias extraídas.
    Caso não consiga encontrar no contexto abaixo ou caso a pergunta não esteja relacionado do contexto jurídico,
    diga apenas "Eu não sei!"
    
    Pergunta: {query}
    
    Contexto: {context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query", "context"]
)


