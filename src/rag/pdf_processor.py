"""
PDF Processor
-------------
Purpose: Process PDF files and extract text.
"""

import os 
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import PyPDF2

logger = logging.getLogger(__name__)    

@staticmethod
def extract_text_pypdf2(pdf_path: str) -> Tuple[str, Dict]:
    """
    Extract text from PDF using PyPDF2.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Tuple of (text, metadata)
        metadata includes: num_pages, title, author (if available)
    
    Note: PyPDF2 works okay for text-based PDFs.
          For scanned PDFs, consider using OCR tools.
    """
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract metadata
            metadata = pdf_reader.metadata or {}
            num_pages = len(pdf_reader.pages)
            
            # Extract text from all pages
            text = ""
            page_texts = {}
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                page_texts[page_num + 1] = page_text
            
            result_metadata = {
                "num_pages": num_pages,
                "title": metadata.get('/Title', 'Unknown'),
                "author": metadata.get('/Author', 'Unknown'),
                "page_texts": page_texts,
                "source_file": os.path.basename(pdf_path)
            }
            
            return text, result_metadata
    
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise


def extract_text_pdfplumber(pdf_path: str) -> Tuple[str, Dict]:
    """
    Extract text from PDF using pdfplumber (better quality).
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Tuple of (text, metadata)
    
    Note: Requires: pip install pdfplumber
          Better text extraction than PyPDF2, especially for complex layouts
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, falling back to PyPDF2")
        return extract_text_pypdf2(pdf_path)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            page_texts = {}
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                page_texts[page_num + 1] = page_text
            
            result_metadata = {
                "num_pages": len(pdf.pages),
                "title": pdf.metadata.get('Title', 'Unknown') if pdf.metadata else 'Unknown',
                "author": pdf.metadata.get('Author', 'Unknown') if pdf.metadata else 'Unknown',
                "page_texts": page_texts,
                "source_file": os.path.basename(pdf_path)
            }
            
            return text, result_metadata
    
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise

class PDFProcessor:
    """
    Process PDF files and extract text for RAG ingestion.
    """
    
    def __init__(self, use_pdfplumber: bool = False):
        """
        Initialize PDF processor.
        
        Args:
            use_pdfplumber: Use pdfplumber (better) or PyPDF2 (built-in)
        """
        self.use_pdfplumber = use_pdfplumber
        
        if use_pdfplumber:
            try:
                import pdfplumber
                logger.info("Using pdfplumber for PDF extraction")
            except ImportError:
                logger.warning("pdfplumber not installed, using PyPDF2")
                self.use_pdfplumber = False
    
    def process_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text from a single PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (extracted_text, metadata)
        
        Example:
            >>> processor = PDFProcessor()
            >>> text, meta = processor.process_pdf("paper.pdf")
            >>> print(f"Extracted {meta['num_pages']} pages")
        """
        pdf_path = str(pdf_path)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        if self.use_pdfplumber:
            text, metadata = extract_text_pdfplumber(pdf_path)
        else:
            text, metadata = extract_text_pypdf2(pdf_path)
        
        logger.info(
            f"✓ Extracted {metadata['num_pages']} pages, "
            f"{len(text)} chars"
        )
        
        return text, metadata
    
    def process_folder(
        self,
        folder_path: str,
        pattern: str = "*.pdf"
    ) -> Dict[str, Tuple[str, Dict]]:
        """
        Process all PDFs in a folder.
        
        Args:
            folder_path: Path to folder containing PDFs
            pattern: File pattern to match (default: "*.pdf")
        
        Returns:
            Dict of {filename: (text, metadata)}
        
        Example:
            >>> processor = PDFProcessor()
            >>> docs = processor.process_folder("./papers")
            >>> for filename, (text, meta) in docs.items():
            ...     print(f"{filename}: {meta['num_pages']} pages")
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        logger.info(f"Processing folder: {folder_path}")
        
        pdf_files = list(folder_path.glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = {}
        failed = []
        
        for pdf_path in pdf_files:
            try:
                text, metadata = self.process_pdf(str(pdf_path))
                documents[pdf_path.stem] = (text, metadata)  # Use filename without extension as key
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                failed.append((pdf_path.name, str(e)))
        
        if failed:
            logger.warning(f"Failed to process {len(failed)} files:")
            for filename, error in failed:
                logger.warning(f"  - {filename}: {error}")
        
        logger.info(f"✓ Processed {len(documents)} PDFs successfully")
        
        return documents
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text (remove extra whitespace, control characters, etc.)
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text
        """
        # Remove multiple newlines
        text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
        
        # Remove control characters (but keep newlines and tabs)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text


# ============ TESTS ============

def test_pdf_processor_missing_file():
    """Test handling of missing file."""
    processor = PDFProcessor()
    
    try:
        processor.process_pdf("nonexistent.pdf")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print("✓ Correctly raises FileNotFoundError for missing file")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    processor = PDFProcessor(use_pdfplumber=False)