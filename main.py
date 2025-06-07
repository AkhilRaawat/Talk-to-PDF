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

# 📦 Load environment variables from .env file
load_dotenv()

# 🔐 Get and set Google API key securely
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# 📊 Function to display PDF statistics
def display_pdf_stats(pdf_docs, text_chunks):
    st.sidebar.markdown("<h4 style='color: #7c4dff;'>📊 PDF Statistics</h4>", unsafe_allow_html=True)
    
    avg_chunk_size = sum(len(chunk) for chunk in text_chunks) // len(text_chunks) if text_chunks else 0
      # Create a more visually appealing stats display
    stats_html = f"""
    <div style="background-color: #f0e6ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #7c4dff; color: #ffffff; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                <span style="font-weight: bold;">{len(pdf_docs)}</span>
            </div>
            <div>Documents</div>
        </div>
        
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #00bcd4; color: #ffffff; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                <span style="font-weight: bold;">{len(text_chunks)}</span>
            </div>
            <div>Total Chunks</div>
        </div>
        
        <div style="display: flex; align-items: center;">
            <div style="background-color: #3f1dcb; color: #ffffff; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                <span style="font-weight: bold;">{avg_chunk_size}</span>
            </div>
            <div>Avg. Chunk Size (chars)</div>
        </div>
    </div>
    """
    
    st.sidebar.markdown(stats_html, unsafe_allow_html=True)

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
        model="gemini-pro", 
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
        vector_db = FAISS.load_local("faiss_index", embeddings)

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
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Initialize session state for processed PDFs flag
    if "pdfs_processed" not in st.session_state:
        st.session_state.pdfs_processed = False
        
    # Initialize session state for PDF metadata
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = []
        
    # Initialize session state for PDF images
    if "pdf_images" not in st.session_state:
        st.session_state.pdf_images = []
          # Initialize session state for document summary
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None
        
    # Set timestamp for reports
    import datetime
    st.session_state.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Page configuration with custom theme
    st.set_page_config(
        page_title="Chat PDF with Gemini",
        page_icon="📚",
        layout="wide", 
        initial_sidebar_state="expanded",        menu_items={
            'Get Help': 'https://github.com/yourusername/talk-to-pdf',
            'Report a bug': 'https://github.com/yourusername/talk-to-pdf/issues',
            'About': 'Chat with PDF using Gemini AI - Analyze and interact with your documents using advanced AI.'
        }
    )
    # Custom CSS for better UI with modern color scheme
    st.markdown("""
        <style>
        /* Modern color palette */
        :root {
            --primary: #7c4dff;
            --primary-light: #b47cff;
            --primary-dark: #3f1dcb;
            --secondary: #00bcd4;
            --secondary-light: #62efff;
            --secondary-dark: #008ba3;
            --background: #f8f9fa;
            --card-bg: #ffffff;
            --text: #212121;
            --text-light: #757575;
        }
        
        /* Main app styling */
        .stApp {
            background-color: var(--background);
        }
        
        /* Headings */
        h1, h2, h3 {
            color: var(--primary-dark);
        }
        
        /* Chat messages */
        .chat-message {
            padding: 1.5rem; 
            border-radius: 0.8rem; 
            margin-bottom: 1.2rem; 
            display: flex;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            background-color: var(--card-bg);
        }
        
        .chat-message.user {
            background-color: #f0e6ff;
            border-left: 5px solid var(--primary);
        }
        
        .chat-message.bot {
            background-color: #e6f9ff;
            border-left: 5px solid var(--secondary);
        }
          /* Buttons */
        .stButton button {
            background-color: var(--primary);
            color: #ffffff;
            border-radius: 8px;
            border: none;
            padding: 10px 15px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            background-color: var(--primary-dark);
            box-shadow: 0 4px 12px rgba(124, 77, 255, 0.25);
            transform: translateY(-2px);
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #fafafa;
        }
        
        /* Input fields */
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 12px;
        }
        
        .stTextInput input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(124, 77, 255, 0.2);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            background-color: #f0f0f0;
        }
          .stTabs [aria-selected="true"] {
            background-color: var(--primary-light) !important;
            color: #ffffff !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: var(--primary-dark);
        }
        
        /* Progress bars */
        .stProgress > div > div > div > div {
            background-color: var(--primary);
        }
        </style>    """, unsafe_allow_html=True)
    
    st.title("📚 Chat with PDF using Gemini 💬")
    st.markdown("<p style='font-size: 1.2rem; color: #757575; margin-top: -10px;'>Upload your PDF documents and ask questions about their content.</p>", unsafe_allow_html=True)
    
    # 📂 Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.markdown("<h3 style='color: #7c4dff;'>📁 Upload PDF(s)</h3>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Select one or more PDF files", accept_multiple_files=True)
        
        # Create columns for buttons
        col1, col2 = st.columns(2)
        with col1:
            process_btn = st.button("📥 Process PDFs", use_container_width=True)
        with col2:
            clear_btn = st.button("🧹 Clear All", use_container_width=True)
            
        if clear_btn:
            # Reset all session state
            st.session_state.chat_history = []
            st.session_state.pdfs_processed = False
            st.session_state.pdf_metadata = []
            st.session_state.pdf_images = []
            st.session_state.document_summary = None
              # Clean up PDF processor if it exists
            if "pdf_processor" in st.session_state:
                st.session_state.pdf_processor.cleanup()
                del st.session_state.pdf_processor
                
            st.rerun()
            
        if process_btn:
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("🔍 Extracting and indexing content..."):
                    # Initialize PDF processor
                    pdf_processor = PDFProcessor()
                    st.session_state.pdf_processor = pdf_processor
                    
                    # Process PDFs for text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("No readable text found in uploaded PDFs.")
                    else:
                        # Process text chunks
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.pdfs_processed = True
                        
                        # Display PDF stats
                        display_pdf_stats(pdf_docs, text_chunks)
                        
                        # Generate document summary if there aren't too many documents
                        if len(pdf_docs) <= 3:  # Limit to avoid overloading
                            with st.spinner("🤖 Generating document summary..."):
                                try:
                                    summary = generate_document_summary(
                                        text_chunks, 
                                        ", ".join([pdf.name for pdf in pdf_docs])
                                    )
                                    st.session_state.document_summary = summary
                                except Exception as e:
                                    st.warning(f"Could not generate summary: {str(e)}")
                        
                        # Extract metadata
                        for pdf in pdf_docs:
                            # Reset file pointer
                            pdf.seek(0)
                            metadata = pdf_processor.get_pdf_metadata(pdf)
                            metadata['summary'] = pdf_processor.generate_pdf_summary(
                                metadata, 
                                len(''.join([chunk for chunk in text_chunks if pdf.name in chunk]))
                            )
                            st.session_state.pdf_metadata.append(metadata)
                            
                            # Extract images (limit to first 3 PDFs to avoid processing too many)
                            if len(st.session_state.pdf_images) < 15:  # Limit total images
                                pdf.seek(0)
                                images = pdf_processor.extract_images_from_pdf(pdf)                             
                                if images:
                                    st.session_state.pdf_images.extend(images[:5])  # Limit 5 images per PDF
                        
                        st.success("✅ PDF content processed and ready for questions!")
        
        # Add feature tabs in sidebar
        if st.session_state.pdfs_processed:
            st.sidebar.divider()
            st.sidebar.markdown("<h3 style='color: #7c4dff;'>📊 PDF Features</h3>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.sidebar.tabs(["📄 Metadata", "🖼️ Images", "📤 Export"])
            
            with tab1:
                # Display PDF metadata
                if st.session_state.pdf_metadata:
                    for meta in st.session_state.pdf_metadata:
                        with st.expander(f"📄 {meta['filename']}"):
                            st.markdown(meta['summary'])
            
            with tab2:
                # Display PDF images if any
                if st.session_state.pdf_images:
                    st.write(f"Found {len(st.session_state.pdf_images)} images in PDFs")
                    if st.button("Show Image Gallery"):
                        st.session_state.pdf_processor.display_image_gallery(
                            st.session_state.pdf_images
                        )
                else:
                    st.info("No images found in the PDFs.")
                    
            with tab3:
                # Export options
                st.subheader("Export Options")
                
                if st.button("📄 Export Chat History"):
                    chat_md = export_chat_history()
                    if chat_md:
                        st.download_button(
                            label="Download Chat History",
                            data=chat_md,
                            file_name="chat_history.md",
                            mime="text/markdown"
                        )
                    else:
                        st.info("No chat history to export.")
                
                if st.button("📊 Export Document Report"):
                    report_md = create_document_report(
                        st.session_state.pdf_metadata, 
                        st.session_state.document_summary
                    )
                    if report_md:
                        st.download_button(
                            label="Download Document Report",                            data=report_md,
                            file_name="document_report.md",
                            mime="text/markdown"
                        )
                    else:
                        st.info("No document data to export.")
        
        # Display app info
        st.sidebar.divider()
        st.sidebar.markdown("<h3 style='color: #7c4dff;'>ℹ️ About</h3>", unsafe_allow_html=True)
        st.sidebar.markdown("""
        <div style="background-color: #f0e6ff; padding: 15px; border-radius: 8px; border-left: 4px solid #7c4dff;">
        This app uses Google's Gemini AI to analyze PDFs and answer your questions.
        <br><br>
        <b>Features:</b>
        <ul style="margin-left: 15px; padding-left: 0;">
          <li>Multi-PDF support</li>
          <li>Semantic search</li>
          <li>Chat history</li>
          <li>Citation of sources</li>          <li>Image extraction</li>
          <li>Export reports</li>
        </ul>
        <br>
        Built with Streamlit, LangChain, and Gemini.
        </div>
        """, unsafe_allow_html=True)

    # Create tabs for different features with custom styling
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Insights", "📝 Summary"])
    
    with tab1:
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
                    answer = user_input(user_question)
                      # Save to chat history
                    save_chat_history(user_question, answer)
                    
                    # Refresh the page to show the new message
                    st.rerun()
        else:
            # Display instruction to process PDFs first
            st.info("👈 Please upload and process PDFs using the sidebar to start chatting.")
    
    with tab2:
        # 📊 Insights tab
        if st.session_state.pdfs_processed:
            st.subheader("📊 Document Insights")
            
            # Show metadata summaries
            if st.session_state.pdf_metadata:
                for meta in st.session_state.pdf_metadata:
                    with st.expander(f"📄 Summary for {meta['filename']}"):
                        st.markdown(meta['summary'])
                        
                        # Add page size visualization if we have multiple pages
                        if len(meta['page_sizes']) > 1:
                            st.subheader("Page Size Distribution")
                            
                            # Create data for visualization
                            page_numbers = list(range(1, len(meta['page_sizes']) + 1))
                            page_areas = [size['width'] * size['height'] for size in meta['page_sizes']]
                            
                            # Create a bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.bar(page_numbers, page_areas)
                            ax.set_xlabel('Page Number')
                            ax.set_ylabel('Page Area (points²)')
                            ax.set_title('Page Size Distribution')
                            
                            # Add a trend line
                            z = np.polyfit(page_numbers, page_areas, 1)
                            p = np.poly1d(z)
                            ax.plot(page_numbers, p(page_numbers), "r--", alpha=0.8)
                            
                            st.pyplot(fig)
            else:
                st.info("No document metadata available yet.")
        else:
            st.info("👈 Please upload and process PDFs using the sidebar to see insights.")
            
    with tab3:
        # 📝 Summary tab
        if st.session_state.pdfs_processed:
            st.subheader("📝 Document Summary")
            
            if st.session_state.document_summary:
                st.markdown(st.session_state.document_summary)
                
                # Add option to regenerate summary
                if st.button("🔄 Regenerate Summary"):
                    with st.spinner("🤖 Regenerating document summary..."):
                        try:
                            # Get text chunks from FAISS index
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                            vector_db = FAISS.load_local("faiss_index", embeddings)
                            
                            # Get sample text for summarization
                            sample_question = "What is this document about?"
                            relevant_chunks = [doc.page_content for doc in vector_db.similarity_search(sample_question)]
                              # Generate new summary
                            summary = generate_document_summary(
                                relevant_chunks, 
                                ", ".join([meta['filename'] for meta in st.session_state.pdf_metadata])
                            )
                            st.session_state.document_summary = summary
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to regenerate summary: {str(e)}")
            else:
                st.info("No document summary available. Try processing fewer or smaller PDFs.")
                
                # Add option to generate summary
                if st.button("✨ Generate Summary"):
                    with st.spinner("🤖 Generating document summary..."):
                        try:
                            # Get text chunks from FAISS index
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                            vector_db = FAISS.load_local("faiss_index", embeddings)
                            
                            # Get sample text for summarization
                            sample_question = "What is this document about?"
                            relevant_chunks = [doc.page_content for doc in vector_db.similarity_search(sample_question)]
                              # Generate new summary
                            summary = generate_document_summary(
                                relevant_chunks, 
                                ", ".join([meta['filename'] for meta in st.session_state.pdf_metadata])
                            )
                            st.session_state.document_summary = summary
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate summary: {str(e)}")
        else:
            st.info("👈 Please upload and process PDFs using the sidebar to see document summary.")

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
