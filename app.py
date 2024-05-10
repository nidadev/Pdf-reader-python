import streamlit as st 
from dotenv import load_dotenv  
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

with st.sidebar:
    st.title("welcome to my sidebar")
    st.text("hello")
    st.markdown('''
        [github]('http://github.com')''')
    
def main():
    st.header("chat with pdf")
    load_dotenv()    

    # upload a file
    pdf = st.file_uploader("upload your pdf file",type='pdf')
     st.write(pdf)
    # st.write(pdf.name)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            #  st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)
     # EMBEDDINGS
        embeddings = OpenAIEmbeddings()
        Vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
        #st.write(Vectorstore)
        store_name = pdf.name[:-4]
        #st.write(store_name)
        #st.write(Vectorstore.index.ntotal)
        #retriever = Vectorstore.as_retriever()

        #  Accept user input
        query = st.text_input("Ask questions about your doc")
        #st.write(query)
        if query:
            doc = Vectorstore.similarity_search(query=query,k=3)
            #st.write(doc)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm,chain_type="stuff")
            response = chain.run(input_documents=doc,question=query)
            st.write(response)
       


if __name__ == '__main__':
    main()
    