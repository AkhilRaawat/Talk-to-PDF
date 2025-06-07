"""
PDF Utilities for enhanced PDF processing
"""

import os
import io
import fitz  # PyMuPDF
import tempfile
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class PDFProcessor:
    """A class to handle advanced PDF processing tasks"""
    
    def __init__(self):
        """Initialize the PDF processor"""
        self.temp_dir = tempfile.mkdtemp()
        
    def extract_images_from_pdf(self, pdf_file, min_width=100, min_height=100) -> List[Dict[str, Any]]:
        """Extract images from a PDF file with metadata"""
        images = []
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.read())
                temp_path = temp_file.name
            
            # Open the PDF with PyMuPDF
            doc = fitz.open(temp_path)
            
            # Iterate through pages
            for page_idx, page in enumerate(doc):
                # Get images
                image_list = page.get_images(full=True)
                
                # Process each image
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Load image with PIL to get dimensions
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    width, height = pil_image.size
                    
                    # Filter small images (likely icons, etc.)
                    if width >= min_width and height >= min_height:
                        # Save image to temp directory
                        img_filename = f"page{page_idx+1}_img{img_idx+1}.{image_ext}"
                        img_path = os.path.join(self.temp_dir, img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Add to results
                        images.append({
                            "path": img_path,
                            "page": page_idx + 1,
                            "width": width,
                            "height": height,
                            "format": image_ext
                        })
            
            # Clean up
            doc.close()
            os.unlink(temp_path)
            
            return images
            
        except Exception as e:
            st.error(f"Error extracting images: {str(e)}")
            return []
    
    def display_image_gallery(self, images: List[Dict[str, Any]], cols=3):
        """Display images in a gallery format"""
        if not images:
            st.info("No images found in the PDF documents.")
            return
        
        # Create figure
        fig = plt.figure(figsize=(15, 5 * (len(images) // cols + 1)))
        
        for i, img_data in enumerate(images):
            # Create subplot
            ax = fig.add_subplot(len(images) // cols + 1, cols, i + 1)
            
            # Display image
            img = mpimg.imread(img_data["path"])
            ax.imshow(img)
            
            # Set title with metadata
            ax.set_title(f"Page {img_data['page']} ({img_data['width']}x{img_data['height']})")
            ax.axis('off')
        
        # Show the plot in Streamlit
        st.pyplot(fig)
    
    def get_pdf_metadata(self, pdf_file) -> Dict[str, Any]:
        """Extract metadata from a PDF file"""
        metadata = {}
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_file.read())
                temp_path = temp_file.name
            
            # Open the PDF with PyMuPDF
            doc = fitz.open(temp_path)
            
            # Get basic metadata
            metadata = {
                "filename": pdf_file.name,
                "page_count": len(doc),
                "title": doc.metadata.get("title", "N/A"),
                "author": doc.metadata.get("author", "N/A"),
                "subject": doc.metadata.get("subject", "N/A"),
                "keywords": doc.metadata.get("keywords", "N/A"),
                "creation_date": doc.metadata.get("creationDate", "N/A"),
                "modification_date": doc.metadata.get("modDate", "N/A"),
                "page_sizes": []
            }
            
            # Get page sizes
            for page in doc:
                metadata["page_sizes"].append({
                    "width": page.rect.width,
                    "height": page.rect.height
                })
            
            # Clean up
            doc.close()
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    def generate_pdf_summary(self, metadata: Dict[str, Any], text_length: int) -> str:
        """Generate a summary of a PDF based on metadata and text length"""
        summary = f"**PDF Summary: {metadata['filename']}**\n\n"
        
        # Add basic info
        summary += f"- **Pages**: {metadata['page_count']}\n"
        summary += f"- **Title**: {metadata['title']}\n"
        summary += f"- **Author**: {metadata['author']}\n"
        
        if metadata['creation_date'] != "N/A":
            # Format the date if available
            creation_date = metadata['creation_date']
            if isinstance(creation_date, str) and creation_date.startswith("D:"):
                # PDF date format conversion
                date_str = creation_date[2:10]  # YYYYMMDD
                summary += f"- **Created**: {date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}\n"
            else:
                summary += f"- **Created**: {creation_date}\n"
        
        # Add text stats
        summary += f"- **Approximate Word Count**: {text_length // 5}\n"  # Rough estimate
        summary += f"- **Approximate Character Count**: {text_length}\n"
        
        return summary
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
