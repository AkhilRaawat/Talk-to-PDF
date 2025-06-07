# 🧠 Import required libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import numpy as np
from pdf_utils import PDFProcessor  # Import our custom PDF utilities

# Set Streamlit page configuration
st.set_page_config(
    page_title="Talk to PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        background-color: white;
        color: #333333;
    }
    .stTextArea > div > div > textarea {
        background-color: white;
        color: #333333;
    }
    .stButton > button {
        background-color: #7c4dff;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #6c3aef;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        color: #1a237e;
    }
    .bot-message {
        background-color: #f3e5f5;
        color: #4a148c;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# 📦 Load environment variables from .env file
load_dotenv()

# 🔐 Get and set Google API key securely
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# 📊 Function to display PDF statistics
def display_pdf_stats(pdf_docs, text_chunks):
    st.sidebar.markdown("### 📊 PDF Statistics", unsafe_allow_html=True)
    
    # Calculate statistics
    num_docs = len(pdf_docs) if pdf_docs else 0
    num_chunks = len(text_chunks) if text_chunks else 0
    avg_chunk_size = sum(len(chunk) for chunk in text_chunks) // len(text_chunks) if text_chunks else 0
    
    # Create three columns for stats
    col1, col2, col3 = st.sidebar.columns(3)
    
    # Documents count
    col1.metric(
        "Documents",
        num_docs,
        help="Number of PDF documents processed"
    )
    
    # Chunks count
    col2.metric(
        "Chunks",
        num_chunks,
        help="Total number of text chunks created"
    )
    
    # Average chunk size
    col3.metric(
        "Avg Size",
        f"{avg_chunk_size:,}",
        help="Average characters per chunk"
    )
    
    # Add a divider
    st.sidebar.markdown("---")

# 💾 Function to save chat history
def save_chat_history(question, answer):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({"question": question, "answer": answer})

# 📝 Function to generate a document summary using Gemini
def generate_document_summary(text_chunks, doc_name="document"):
    """Generate a comprehensive summary of the document using Gemini AI"""
    try:
        # Create a prompt for the summary
        summary_prompt = f"""
        You are an expert document summarizer. Create a comprehensive summary of the following document: {doc_name}.
        Focus on the main themes, key points, and important information.
        
        Format the summary with appropriate headings, bullet points for key information, and include a brief overview at the beginning.
        
        Document content:
        {' '.join(text_chunks[:5])}  # Using first few chunks to get the general content
        """
        
        # Use Gemini to generate the summary
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(summary_prompt)
        
        return response.text
    except Exception as e:
        return f"Failed to generate summary: {str(e)}"

# 📊 Function to export chat history as markdown
def export_chat_history():
    """Export the chat history as a markdown file"""
    if not st.session_state.chat_history:
        return None
    
    md_content = "# Chat with PDF - Conversation History\n\n"
    md_content += f"*Generated on {st.session_state.get('timestamp', 'N/A')}*\n\n"
    
    for i, message in enumerate(st.session_state.chat_history):
        md_content += f"## Question {i+1}\n\n"
        md_content += f"**User:** {message['question']}\n\n"
        md_content += f"**Gemini:** {message['answer']}\n\n"
        md_content += "---\n\n"
    
    return md_content

# 📄 Function to create a document report
def create_document_report(pdf_metadata, summary=None):
    """Create a comprehensive document report including metadata and summary"""
    if not pdf_metadata:
        return None
    
    report = "# Document Analysis Report\n\n"
    report += f"*Generated on {st.session_state.get('timestamp', 'N/A')}*\n\n"
    
    for meta in pdf_metadata:
        report += f"## {meta['filename']}\n\n"
        report += f"- **Pages:** {meta['page_count']}\n"
        report += f"- **Title:** {meta.get('title', 'N/A')}\n"
        report += f"- **Author:** {meta.get('author', 'N/A')}\n"
        report += f"- **Created:** {meta.get('creation_date', 'N/A')}\n\n"
        
        if 'summary' in meta:
            report += "### Document Summary\n\n"
            report += f"{meta['summary']}\n\n"
        
    if summary:
        report += "## AI-Generated Content Summary\n\n"
        report += f"{summary}\n\n"
    
    return report

# 📄 Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    full_text = ""
    total_pages = 0
    processed_pages = 0
    failed_pdfs = []
    
    # Create a progress bar
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # First pass to count total pages
        for pdf in pdf_docs:
            try:
                reader = PdfReader(pdf)
                total_pages += len(reader.pages)
                # Reset file pointer
                pdf.seek(0)
            except Exception:
                # Just count this as unknown for now
                pass
        
        # Second pass to actually process
        for pdf_idx, pdf in enumerate(pdf_docs):
            try:
                status_text.text(f"Processing: {pdf.name}")
                reader = PdfReader(pdf)
                
                for page_idx, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Add page and document metadata as context
                            full_text += f"\n--- Document: {pdf.name}, Page: {page_idx+1} ---\n"
                            full_text += page_text + "\n"
                        processed_pages += 1
                        
                        # Update progress
                        if total_pages > 0:
                            progress_bar.progress(processed_pages / total_pages)
                    except Exception as e:
                        st.sidebar.warning(f"Skipped page {page_idx+1} in {pdf.name}: {str(e)}")
                        
            except Exception as e:
                failed_pdfs.append(pdf.name)
                st.sidebar.warning(f"Failed to process {pdf.name}: {str(e)}")
                
        # Finalize progress
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Report on failures if any
        if failed_pdfs:
            st.sidebar.error(f"Failed to process {len(failed_pdfs)} PDFs: {', '.join(failed_pdfs)}")
            
        return full_text
    
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred: {str(e)}")
        return full_text

# ✂️ Function to split long text into smaller chunks (important for context)
def get_text_chunks(text):
    # Using RecursiveCharacterTextSplitter to ensure semantic chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Smaller chunks for more precise retrieval
        chunk_overlap=150,  # Some overlap to maintain context between chunks
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioritize splitting at paragraph/sentence boundaries
        length_function=len
    )
    chunks = splitter.split_text(text)
    
    # Show progress
    st.sidebar.text(f"Created {len(chunks)} text chunks")
    return chunks

# 🧠 Function to convert chunks into embeddings and store in FAISS index
def get_vector_store(text_chunks):
    try:
        with st.sidebar.status("Creating vector embeddings...") as status:
            # More advanced embeddings with metadata
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Add chunk index as metadata
            texts_with_metadata = []
            for i, chunk in enumerate(text_chunks):
                # Extract document and page info if available
                doc_info = "Unknown document"
                page_info = "Unknown page"
                
                # Look for the metadata we added during text extraction
                if "--- Document:" in chunk:
                    meta_line = chunk.split("\n")[0]
                    if "Page:" in meta_line:
                        try:
                            doc_part = meta_line.split("Document:")[1].split(",")[0].strip()
                            page_part = meta_line.split("Page:")[1].strip()
                            doc_info = doc_part
                            page_info = page_part
                        except:
                            pass
                
                # Create metadata for this chunk
                metadata = {
                    "chunk_id": i,
                    "document": doc_info,
                    "page": page_info,
                    "chunk_size": len(chunk)
                }
                
                texts_with_metadata.append((chunk, metadata))
            
            # Create FAISS index with metadata
            vector_store = FAISS.from_texts(
                [text for text, _ in texts_with_metadata],
                embedding=embeddings,
                metadatas=[metadata for _, metadata in texts_with_metadata]
            )
            
            # Save to disk for reuse
            vector_store.save_local("faiss_index")
            status.update(label="Vector store created successfully!", state="complete")
            
            return vector_store
    except Exception as e:
        st.sidebar.error(f"Error creating vector store: {str(e)}")
        raise

# 🔄 Create a QA chain using Gemini + custom prompt
def get_conversational_chain():
    # Enhanced prompt to guide the model's behavior and restrict hallucination
    prompt_template = """
    You are an intelligent assistant that helps users understand PDF documents.
    
    Answer the question as detailed as possible based ONLY on the provided context.
    If the answer is not available in the context, say: "I don't have enough information to answer this question based on the PDF content."
    Always maintain a helpful, friendly tone.
    
    Try to include relevant quotes from the document to support your answer.
    For each piece of information you use, include a citation with the document name and page number in parentheses at the end of the sentence.
    
    If the question is ambiguous or you need more clarity, guide the user to ask a more specific question.
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Gemini model initialization with parameters tuned for QA
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Using the faster Gemini 1.5 Flash model
        temperature=0.2,  # Lower temperature for more factual responses
        top_p=0.95,       # Control token selection diversity
        top_k=40,         # Consider top 40 tokens for diversity without going off-topic
        max_output_tokens=2048  # Allow for longer, more comprehensive answers
    )
    
    # Load QA chain using 'stuff' method (puts all context in one prompt)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# 🗣️ Function to handle user query
def user_input(user_question):
    try:
        # Load saved FAISS index with the same embedding model used before
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.load_local(
            "faiss_index", 
            embeddings,
            allow_dangerous_deserialization=True  # Only safe because we created this index ourselves
        )

        # Enhanced retrieval with more relevant chunks and metadata
        relevant_docs = vector_db.similarity_search_with_score(
            user_question,
            k=5  # Retrieve top 5 most relevant chunks
        )
        
        # Filter out low relevance chunks (high distance scores)
        filtered_docs = [doc for doc, score in relevant_docs if score < 0.8]
        
        # Log retrieval results (for debugging)
        source_documents = []
        for doc, score in relevant_docs:
            source = f"{doc.metadata.get('document', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}"
            if source not in source_documents:
                source_documents.append(source)
        
        # Get QA chain and run it with context + user question
        qa_chain = get_conversational_chain()
        response = qa_chain({
            "input_documents": filtered_docs,
            "question": user_question
        }, return_only_outputs=True)
        
        answer = response["output_text"]
        
        # Add sources information if not already included
        if not any(src in answer for src in source_documents) and source_documents:
            answer += "\n\n**Sources:**\n" + "\n".join([f"- {src}" for src in source_documents])
            
        return answer
    
    except Exception as e:
        return f"An error occurred while processing your question: {str(e)}"

# 🧭 Main Streamlit app
def main():
    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdfs_processed" not in st.session_state:
        st.session_state.pdfs_processed = False
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = []
    if "pdf_images" not in st.session_state:
        st.session_state.pdf_images = []
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    
    # Set timestamp for reports
    import datetime
    st.session_state.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Main page header
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #7c4dff;'>📚 Talk to Your PDF</h1>
            <p style='color: #666666; font-size: 1.2em;'>Upload your PDFs and chat with them using AI</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.markdown("<h3 style='color: #7c4dff;'>📄 Upload Your Documents</h3>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Upload your PDFs here",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files"
        )

        if st.button("Process PDFs", type="primary"):
            if not pdf_docs:
                st.error("⚠️ Please upload at least one PDF file before processing.")
                return
            
            try:
                with st.spinner("Processing your PDFs..."):
                    # Get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("⚠️ No readable text found in the uploaded PDFs. Please make sure the PDFs contain text content.")
                        return
                    
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("⚠️ Could not process the PDF content. Please try with different PDF files.")
                        return
                    
                    # Create vector store
                    try:
                        vector_store = get_vector_store(text_chunks)
                        st.session_state.pdfs_processed = True
                        # Display PDF statistics
                        display_pdf_stats(pdf_docs, text_chunks)
                        st.success("✅ PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"⚠️ Error creating vector store: {str(e)}")
                        return
                    
            except Exception as e:
                st.error(f"⚠️ Error processing PDFs: {str(e)}")
                return

    # Main chat interface
    if st.session_state.pdfs_processed:
        # Chat interface
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["type"] == "user":
                st.markdown(f"""
                    <div class='chat-message user'>
                        <div>
                            <b>You:</b><br>
                            {message["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='chat-message bot'>
                        <div>
                            <b>AI:</b><br>
                            {message["content"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # User input
        def submit_question():
            st.session_state.question_submitted = True

        question = st.text_input(
            "Ask a question about your documents:",
            key="question_input",
            on_change=submit_question
        )

        if "question_submitted" in st.session_state and st.session_state.question_submitted:
            if question and question != st.session_state.user_question:
                st.session_state.user_question = question
                with st.spinner("Thinking..."):
                    answer = user_input(question)
                    
                    # Save to chat history
                    st.session_state.chat_history.append({"type": "user", "content": question})
                    st.session_state.chat_history.append({"type": "bot", "content": answer})
                
                st.session_state.question_submitted = False
                st.rerun()

        # Export options
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Chat History", type="secondary"):
                chat_export = export_chat_history()
                if chat_export:
                    st.download_button(
                        "Download Chat History",
                        chat_export,
                        file_name="chat_history.md",
                        mime="text/markdown"
                    )
        
        with col2:
            if st.button("Generate Report", type="secondary"):
                report = create_document_report(
                    st.session_state.pdf_metadata,
                    st.session_state.document_summary
                )
                if report:
                    st.download_button(
                        "Download Report",
                        report,
                        file_name="document_report.md",
                        mime="text/markdown"
                    )

    else:
        # Welcome message when no PDFs are processed
        st.markdown("""
            <div style='text-align: center; padding: 2rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #7c4dff;'>👋 Welcome!</h3>
                <p style='color: #666666;'>Upload your PDF documents using the sidebar to get started.</p>
                <p style='color: #666666; font-size: 0.9em;'>You can upload multiple PDFs and chat with them using AI.</p>
            </div>
        """, unsafe_allow_html=True)

# 🚀 Run the app
if __name__ == "__main__":
    main()
      # Set up custom footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 12px; background-color: #7c4dff; color: #ffffff; font-size: 14px; box-shadow: 0 -2px 10px rgba(0,0,0,0.1);">
        Built with ❤️ using Streamlit and Gemini AI | © 2025
    </div>
    """, unsafe_allow_html=True)
    
    # Add keyboard shortcuts for convenience
    st.markdown("""
    <script>
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to submit question
        if (e.ctrlKey && e.key === 'Enter') {
            document.querySelector('input[type="text"]').focus();
        }
    });
    </script>
    """, unsafe_allow_html=True)
