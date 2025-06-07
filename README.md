# 📚 Talk to PDF with Gemini AI

An interactive application that allows users to upload PDF documents and have natural conversations about their content using Google's Gemini AI.

## 🌟 Features

- **Multi-PDF Support**: Upload and process multiple PDFs at once
- **Smart Chunking**: Intelligently breaks documents into semantic chunks for better context understanding
- **Vector Search**: Uses FAISS and Google embeddings for fast similarity search
- **Citations**: Provides document and page citations for answers
- **Chat Interface**: Clean, user-friendly chat interface with history
- **Document Statistics**: Provides insights about the processed documents

## 🛠️ Technology Stack

- **Streamlit**: For the web interface
- **LangChain**: For orchestrating the AI workflow
- **Google Gemini AI**: For generating embeddings and answering questions
- **FAISS**: For vector storage and similarity search
- **PyPDF2**: For PDF text extraction

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the App

```
streamlit run main.py
```

## 📋 Usage Guide

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Process Documents**: Click the "Process PDFs" button to extract and index the content
3. **Ask Questions**: Type your questions in the input field at the bottom of the chat interface
4. **View Answers**: See Gemini's responses with citations to the source documents
5. **Clear Chat**: Use the "Clear All" button to reset the chat and upload new documents

## 🔍 How It Works

1. **Text Extraction**: The app extracts text from PDFs using PyPDF2
2. **Text Chunking**: The text is divided into smaller, semantically meaningful chunks
3. **Embedding Generation**: Each chunk is converted into a vector embedding using Google's embedding model
4. **Vector Storage**: Embeddings are stored in a FAISS index for efficient similarity search
5. **Query Processing**: When you ask a question, the app finds the most relevant chunks
6. **Answer Generation**: Gemini AI generates comprehensive answers based on the relevant context

## 🔒 Privacy

- All processing happens on your local machine
- PDF content is not stored permanently
- No data is sent to external servers except to Google's API for embeddings and text generation

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.


