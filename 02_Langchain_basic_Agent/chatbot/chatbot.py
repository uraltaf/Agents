import streamlit as st
import dotenv
from langchain.schema.messages import HumanMessage, SystemMessage
# from langchain_intro.chatbot import chat_model

from langchain.prompts import (ChatPromptTemplate , PromptTemplate , SystemMessagePromptTemplate, HumanMessagePromptTemplate) 


dotenv.load_dotenv()
from langchain_groq import ChatGroq

chat_model = ChatGroq (model = "llama-3.1-70b-versatile")

# messages = [

#     SystemMessage(
        
#         content = """ You're an assistant knowledeable about 
#         healthcare. Only answer health-care related questions"""
#     ),
#     HumanMessage(content = "What is the AI?"),
#     ]

# result = chat_model.invoke(messages)
# print(result)

review_template_str = """ Your job is to use patient reviews to answer questions about
their experience at a hospital. Use the following context to 
answer questions. Be a detailed as possible, but don't make up
any information that's not from the context. If you don't know an answer, say you don't know.

{context}

{question}
"""

review_template = ChatPromptTemplate.from_template(review_template_str)

context = "I had a great stay"

question = "Did anyone have a positive experience"

formatted_prompt = review_template.format(context=context, question=question)

messages = [
    HumanMessage(content= formatted_prompt)
]

result = chat_model.invoke(messages)

print(result)

