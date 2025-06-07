"""
Advanced document analysis features for Talk to PDF application.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from typing import List, Dict, Any, Tuple, Optional

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class DocumentAnalyzer:
    """A class for performing advanced document analysis"""
    
    def __init__(self):
        """Initialize the document analyzer"""
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['said', 'would', 'also', 'could', 'may', 'one', 'two', 'many'])
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, punctuation, and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_word_cloud(self, text: str, title: str = "Word Cloud") -> plt.Figure:
        """Generate a word cloud visualization from text"""
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Tokenize
        words = word_tokenize(clean_text)
        
        # Remove stopwords
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=150,
            contour_width=1
        ).generate(' '.join(filtered_words))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        return fig
    
    def extract_top_keywords(self, text: str, n: int = 20) -> pd.DataFrame:
        """Extract the top keywords from text"""
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        # Tokenize
        words = word_tokenize(clean_text)
        
        # Remove stopwords
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Convert to DataFrame
        df = pd.DataFrame(word_counts.most_common(n), columns=['Word', 'Frequency'])
        
        return df
    
    def plot_keyword_distribution(self, text: str, n: int = 15, title: str = "Top Keywords") -> plt.Figure:
        """Plot distribution of top keywords"""
        # Get top keywords
        df = self.extract_top_keywords(text, n)
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bars with custom color gradient
        bars = ax.barh(df['Word'], df['Frequency'], color=plt.cm.viridis(np.linspace(0, 0.8, len(df))))
        
        # Add value labels to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                   f"{width}", ha='left', va='center', fontweight='bold')
        
        # Customize chart appearance
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Invert y-axis to have highest frequency at the top
        ax.invert_yaxis()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def analyze_document_content(self, text_chunks: List[str]) -> Dict[str, Any]:
        """Perform comprehensive document analysis"""
        # Combine all text chunks
        full_text = ' '.join(text_chunks)
        
        # Basic text statistics
        word_count = len(re.findall(r'\b\w+\b', full_text))
        sentence_count = len(re.findall(r'[.!?]+', full_text))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Extract keyword distribution
        keywords_df = self.extract_top_keywords(full_text, 50)
        
        # Calculate reading time (average reading speed is ~250 words per minute)
        reading_time_minutes = word_count / 250
        
        # Compile results
        results = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "reading_time_minutes": reading_time_minutes,
            "keywords": keywords_df,
            "full_text": full_text
        }
        
        return results
    
    def generate_document_insights(self, text_chunks: List[str], title: str = "Document Analysis") -> Dict[str, Any]:
        """Generate visual insights about document content"""
        # Get analysis results
        analysis = self.analyze_document_content(text_chunks)
        
        # Generate visualizations
        wordcloud_fig = self.generate_word_cloud(analysis["full_text"], title=f"Word Cloud: {title}")
        keyword_fig = self.plot_keyword_distribution(analysis["full_text"], title=f"Top Keywords: {title}")
        
        # Add figures to results
        analysis["wordcloud_fig"] = wordcloud_fig
        analysis["keyword_fig"] = keyword_fig
        
        return analysis
    
    def display_document_insights(self, insights: Dict[str, Any]) -> None:
        """Display document insights in Streamlit"""
        # Display text statistics
        st.subheader("📊 Document Statistics")
        
        # Create metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", f"{insights['word_count']:,}")
        col2.metric("Sentence Count", f"{insights['sentence_count']:,}")
        col3.metric("Reading Time", f"{insights['reading_time_minutes']:.1f} min")
        
        # Display average sentence length
        st.info(f"📏 Average Sentence Length: {insights['avg_sentence_length']:.1f} words")
        
        # Display wordcloud
        st.subheader("🔤 Word Cloud")
        st.pyplot(insights["wordcloud_fig"])
        
        # Display keyword distribution
        st.subheader("🔑 Keyword Distribution")
        st.pyplot(insights["keyword_fig"])
        
        # Display top keywords as a table
        st.subheader("📋 Top Keywords")
        st.dataframe(insights["keywords"].head(20), use_container_width=True)
