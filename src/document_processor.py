import os
import json
import base64
from pathlib import Path
from typing import Union, List, Optional, Tuple
from datetime import datetime
import mimetypes

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    from PIL import Image
except ImportError:
    Image = None

from .models import DocumentMetadata


class DocumentProcessor:
    """Process various document formats and extract text/images"""

    SUPPORTED_FORMATS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".txt"}

    def __init__(self, output_dir: str = "extracted_content"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True)

    def process_document(
        self, file_path: str, extract_images: bool = True
    ) -> Tuple[str, DocumentMetadata]:
        """
        Process document and extract text

        Args:
            file_path: Path to document
            extract_images: Whether to extract and process images

        Returns:
            Tuple of (extracted_text, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_type = file_path.suffix.lower()
        if file_type not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file type: {file_type}")

        start_time = datetime.now()

        if file_type == ".pdf":
            text, pages = self._process_pdf(file_path, extract_images)
        elif file_type == ".txt":
            text = file_path.read_text(encoding="utf-8")
            pages = 1
        elif file_type in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            text = self._process_image(file_path)
            pages = 1
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        processing_time = (datetime.now() - start_time).total_seconds()

        metadata = DocumentMetadata(
            file_path=str(file_path),
            file_type=file_type.lstrip("."),
            file_size=file_path.stat().st_size,
            pages=pages,
            extraction_method="PyPDF2" if file_type == ".pdf" else "OCR",
            extracted_at=datetime.now(),
            processing_time_seconds=processing_time,
        )

        return text, metadata

    def _process_pdf(self, file_path: Path, extract_images: bool = True) -> Tuple[str, int]:
        """Extract text from PDF"""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

        text_parts = []
        
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

            return "\n\n".join(text_parts), num_pages
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

    def _process_image(self, file_path: Path) -> str:
        """Process image file (returns base64 for API submission)"""
        if Image is None:
            raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")

        try:
            with open(file_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"[IMAGE_DATA:{file_path.suffix}:{img_data}]"
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")

    def extract_images_from_pdf(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """Extract images from PDF"""
        if convert_from_path is None:
            raise ImportError("pdf2image is required. Install with: pip install pdf2image")

        output_dir = output_dir or self.output_dir
        Path(output_dir).mkdir(exist_ok=True)

        try:
            images = convert_from_path(file_path)
            saved_paths = []

            for i, image in enumerate(images):
                output_path = Path(output_dir) / f"page_{i+1}.png"
                image.save(output_path, "PNG")
                saved_paths.append(str(output_path))

            return saved_paths
        except Exception as e:
            raise RuntimeError(f"Failed to extract images from PDF: {str(e)}")
