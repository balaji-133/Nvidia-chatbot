# import streamlit as st
# import os
# import time

# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate

# # --- FIX: SWITCH TO THE UNIVERSAL 'RetrievalQA' CHAIN ---
# # This import works on almost ALL versions of LangChain, bypassing your error.
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv

# load_dotenv()

# # Ensure API Key is loaded
# os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

# # Initialize the LLM
# llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1.5")

# def vector_embeddings():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = NVIDIAEmbeddings()
#         # Ensure 'us_census' folder exists with PDFs inside
#         st.session_state.loader = PyPDFDirectoryLoader("./us_census")
#         st.session_state.docs = st.session_state.loader.load()
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# st.title("US Census Report Chatbot")

# # For RetrievalQA, we use a specific prompt variable name usually
# prompt_template = """
# Answer the question based on the context below.
# Please provide the most accurate response based on the US Census Report.
# <context>
# {context}
# </context>
# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(prompt_template)

# prompt1 = st.text_input("Enter your question about US Census Report:")

# if st.button("Document embedding"):
#     with st.spinner("Creating Vector Store..."):
#         vector_embeddings()
#         st.success("Vector Store DB is ready using NVIDIA embeddings")

# if prompt1:
#     if "vectors" not in st.session_state:
#         st.error("Please click 'Document embedding' first to load the documents.")
#     else:
#         # --- LOGIC UPDATE FOR STABILITY ---
#         # We use RetrievalQA.from_chain_type which is much more robust against version mismatches
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=st.session_state.vectors.as_retriever(),
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": prompt}
#         )
        
#         start = time.process_time()
#         # RetrievalQA expects the input key to be "query"
#         response = qa_chain.invoke({"query": prompt1})
        
#         print("Time taken to fetch response:", time.process_time() - start)
        
#         # In RetrievalQA, the answer is in 'result', not 'answer'
#         st.write(response['result'])

#         with st.expander("View context"):
#             # In RetrievalQA, sources are in 'source_documents'
#             for i, doc in enumerate(response["source_documents"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------")


import streamlit as st
import os
import tempfile
import time

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# --- THE CORRECT IMPORT ---
# This requires the 'langchain' package to be in requirements.txt
# Source - https://stackoverflow.com/a
# Posted by Bazzas
# Retrieved 2026-01-17, License - CC BY-SA 4.0

from langchain.chains.retrieval_qa.base import RetrievalQA


from dotenv import load_dotenv

# 1. Page Configuration (Must be the first line of Streamlit command)
st.set_page_config(page_title="NVIDIA DocuChat", layout="wide", page_icon="🤖")

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("NVIDIA_API_KEY"):
    st.warning("⚠️ NVIDIA_API_KEY not found in environment variables. Please check your .env file.")

# 2. Initialize LLM (Using a stable model to prevent warnings)
# You can switch this back to 'nvidia/llama-3.3-nemotron-super-49b-v1.5' if you prefer
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# 3. Define the Embedding Function
def process_pdf(uploaded_file):
    """
    Handles the logic of saving the uploaded file, loading it, 
    splitting it, and creating the vector store.
    """
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(docs)

        # Create Embeddings
        embeddings = NVIDIAEmbeddings()
        
        # Create Vector Store
        vectors = FAISS.from_documents(final_documents, embeddings)
        return vectors
    
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)

# 4. Sidebar UI for File Upload
with st.sidebar:
    st.image("https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-p@2x.png", width=150)
    st.title("📄 Document Settings")
    st.write("Upload a PDF to chat with it using NVIDIA AI.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    process_button = st.button("🚀 Process Uploaded PDF")

    if process_button and uploaded_file:
        with st.spinner("Processing PDF... (This may take a moment)"):
            try:
                st.session_state.vectors = process_pdf(uploaded_file)
                st.success("✅ PDF Processed! You can now ask questions.")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    st.markdown("---")
    st.caption("Powered by LangChain & NVIDIA NIM")

# 5. Main Chat Interface
st.title("🤖 NVIDIA AI DocuChat")
st.markdown("Ask questions about your uploaded PDF document.")

# Initialize chat history in session state if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Logic
prompt_input = st.chat_input("Ask a question about your document...")

if prompt_input:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # Check if vectors are ready
    if "vectors" not in st.session_state:
        response_text = "⚠️ Please upload and process a PDF file in the sidebar first."
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.process_time()
                
                # Define the Prompt
                prompt_template = """
                Answer the question based ONLY on the context below. 
                If the answer is not in the context, say "I don't know".
                
                <context>
                {context}
                </context>
                
                Question: {question}
                """
                prompt = ChatPromptTemplate.from_template(prompt_template)
                
                # Create the Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectors.as_retriever(),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
                
                # Get Response
                response = qa_chain.invoke({"query": prompt_input})
                response_text = response['result']
                
                st.markdown(response_text)
                
                # Optional: Show search time
                time_taken = time.process_time() - start
                st.caption(f"⏱️ Response generated in {time_taken:.2f} seconds")

                # Show sources in an expander
                with st.expander("🔍 View Context / Sources"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.write(doc.page_content)
                        st.divider()

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
