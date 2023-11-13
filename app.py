import os
import pickle
import time
from flask import Flask, render_template, request
from langchain import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__) 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/urls', methods=['GET', 'POST'])
def urls():
    if request.method == 'POST':
        urls = [
            request.form.get('url1'),
            request.form.get('url2'),
            request.form.get('url3')
        ]

        llm = OpenAI(temperature=0.9, max_tokens=500)

        if request.form.get('process_url_clicked'):
            loader = UnstructuredURLLoader(urls=urls)

            # Data Loading
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            # Create embeddings and save to FAISS index
            embeddings = HuggingFaceEmbeddings(model_name="deepset/sentence_bert")
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            # Save the FAISS index to a pickle file
            file_path = "faiss_store_url.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

        query = request.form.get('question')
        if query:
            file_path = "faiss_store_url.pkl"
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever()
                    )
                    result = chain({"question": query}, return_only_outputs=True)
                    answer = result["answer"]
                    sources = result.get("sources", "").split("\n")

                    return render_template(
                        'index.html',
                        answer=answer,
                        sources=sources
                    )

    return render_template('index.html')

@app.route('/pdf', methods=['GET','POST'])
def pdf():
    if request.method == 'POST':
        llm = OpenAI(temperature=0.9, max_tokens=500)
        
        if 'process_pdf_clicked' in request.form:

            # Check if the file is present in the request
            if 'pdf' in request.files:
                pdf = request.files['pdf']

                # Ensure the file has a valid filename
                if pdf.filename.endswith('.pdf'):
                    text = ""
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )

                    chunks = text_splitter.split_text(text)

                    embeddings = HuggingFaceEmbeddings(model_name="deepset/sentence_bert")
                    knowledge_base = FAISS.from_texts(chunks, embeddings)

                    file_path = "faiss_store_pdf.pkl"
                    with open(file_path, "wb") as f:
                        pickle.dump(knowledge_base, f)

                else:
                    return "Invalid file format. Please upload a PDF file."

            else:
                return "No PDF file uploaded."
        user_question = request.form.get('user_question')
        if user_question:
            file_path = "faiss_store_pdf.pkl"
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                docs = vectorstore.similarity_search(user_question)

                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                    return render_template("query.html", response=response, user_question=user_question)

    return render_template("query.html")


if __name__ == '__main__':
    app.run()
