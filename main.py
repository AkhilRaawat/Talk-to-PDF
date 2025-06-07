# 🧠 Import required libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os

# 📦 Load environment variables from .env file
load_dotenv()

# 🔐 Get and set Google API key securely
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# 📄 Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    full_text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text
        except Exception as e:
            st.warning(f"Failed to read {pdf.name}: {str(e)}")
    return full_text

# ✂️ Function to split long text into smaller chunks (important for context)
def get_text_chunks(text):
    # Using RecursiveCharacterTextSplitter to ensure sentences aren't broken mid-way
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# 🧠 Function to convert chunks into embeddings and store in FAISS index
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save to disk for reuse

# 🔄 Create a QA chain using Gemini + custom prompt
def get_conversational_chain():
    # Prompt to guide the model's behavior and restrict hallucination
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not available in the context, say: "Answer is not available in the context".
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Gemini model initialization with low temperature for consistent answers
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Load QA chain using 'stuff' method (puts all context in one prompt)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# 🗣️ Function to handle user query
def user_input(user_question):
    # Load saved FAISS index with the same embedding model used before
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.load_local("faiss_index", embeddings)

    # Search for the most relevant chunks to the user's question
    relevant_docs = vector_db.similarity_search(user_question)

    # Get QA chain and run it with context + user question
    qa_chain = get_conversational_chain()
    response = qa_chain({
        "input_documents": relevant_docs,
        "question": user_question
    }, return_only_outputs=True)

    # Display the final answer from Gemini
    st.markdown("### 🤖 Gemini's Response:")
    st.write(response["output_text"])

# 📊 Function to display PDF statistics
def display_pdf_stats(pdf_docs, text_chunks):
    st.sidebar.subheader("📊 PDF Statistics")
    st.sidebar.info(f"📄 Documents: {len(pdf_docs)}\n"
                   f"📝 Total Chunks: {len(text_chunks)}\n"
                   f"📏 Avg. Chunk Size: {sum(len(chunk) for chunk in text_chunks) // len(text_chunks) if text_chunks else 0} chars")

# 💾 Function to save chat history
def save_chat_history(question, answer):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({"question": question, "answer": answer})

# 🧭 Main Streamlit app
def main():
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Initialize session state for processed PDFs flag
    if "pdfs_processed" not in st.session_state:
        st.session_state.pdfs_processed = False
        
    # Page configuration with custom theme
    st.set_page_config("Chat PDF", layout="wide", 
                      page_icon="📚",
                      initial_sidebar_state="expanded")
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .chat-message {
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem; 
            display: flex;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chat-message.user {
            background-color: #e6f3ff;
            border-left: 5px solid #2e86de;
        }
        .chat-message.bot {
            background-color: #f0f7f2;
            border-left: 5px solid #26c281;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 Chat with PDF using Gemini 💬")
    st.markdown("Upload your PDF documents and ask questions about their content.")

    # 📂 Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.header("📁 Upload PDF(s)")
        pdf_docs = st.file_uploader("Select one or more PDF files", accept_multiple_files=True)
        
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("📥 Process PDFs", use_container_width=True)
        with col2:
            clear_btn = st.button("🧹 Clear All", use_container_width=True)
            
        if clear_btn:
            # Reset all session state
            st.session_state.chat_history = []
            st.session_state.pdfs_processed = False
            st.experimental_rerun()
            
        if process_btn:
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("🔍 Extracting and indexing content..."):
                    # Process PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found in uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.pdfs_processed = True
                        # Display PDF stats
                        display_pdf_stats(pdf_docs, text_chunks)
                        st.success("✅ PDF content processed and ready for questions!")
        
        # Display app info
        st.sidebar.divider()
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.markdown("""
        This app uses Google's Gemini AI to analyze PDFs and answer your questions.
        
        **Features:**
        - Multi-PDF support
        - Semantic search
        - Chat history
        - Citation of sources
        
        Built with Streamlit, LangChain, and Gemini.
        """)

    # 💬 Chat interface
    if st.session_state.pdfs_processed:
        # Display chat messages
        for i, message in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message user">
                <div>
                    <b>You:</b><br>{message["question"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="chat-message bot">
                <div>
                    <b>Gemini:</b><br>{message["answer"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        # Input for new question
        user_question = st.text_input("Ask a question about your documents:", key="user_input")
        
        if user_question:
            # Get the model's response
            with st.spinner("Thinking..."):
                # Load FAISS index
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_db = FAISS.load_local("faiss_index", embeddings)

                # Search for relevant chunks
                relevant_docs = vector_db.similarity_search(user_question)

                # Get answer
                qa_chain = get_conversational_chain()
                response = qa_chain({
                    "input_documents": relevant_docs,
                    "question": user_question
                }, return_only_outputs=True)
                
                answer = response["output_text"]
                
                # Save to chat history
                save_chat_history(user_question, answer)
                
                # Refresh the page to show the new message
                st.experimental_rerun()
    else:
        # Display instruction to process PDFs first
        st.info("👈 Please upload and process PDFs using the sidebar to start chatting.")

# 🚀 Run the app
if __name__ == "__main__":
    main()
