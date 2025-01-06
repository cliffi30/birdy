import re
from typing import List, Tuple, Dict

import fitz  # PyMuPDF
import pandas as pd
from langchain.docstore.document import Document


def extract_structured_content(pdf_path: str, num_pages: int = None) -> List[Document]:
    """
    Extract structured content from PDF including tables, text blocks, and images.
    Returns a list of LangChain Document objects with appropriate metadata.
    """
    doc = fitz.open(pdf_path)
    documents = []

    if num_pages is None:
        num_pages = len(doc)
    for page_num in range(num_pages):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # Text block
                spans = block.get("lines", [])
                block_text = " ".join([
                    span["text"] for line in spans 
                    for span in line.get("spans", []) 
                    if "text" in span
                ])

                if _is_unwanted_text(block_text):
                    continue

                bbox = block["bbox"]
                first_line_spans = spans[0]["spans"] if spans else []
                is_title = _is_emphasized_text(first_line_spans)

                if is_title:
                    block_text = block_text.title()

                documents.append(Document(
                    page_content=block_text,
                    metadata={
                        "bbox": bbox,
                        "is_title": is_title,
                        "type": "title" if is_title else "text",
                        "page_number": page_num
                    }
                ))

            elif block.get("type") == 1:  # Image block
                try:
                    if all(k in block for k in ["ext", "width", "height", "colorspace", "image"]):
                        image_ext = block["ext"].lower()
                        if image_ext not in ["jpg", "jpeg", "png"]:
                            continue

                        nearby_text = " ".join([
                            doc.metadata["page_content"] for doc in documents 
                            if "bbox" in doc.metadata and _is_nearby(block["bbox"], doc.metadata["bbox"])
                        ])

                        documents.append(Document(
                            page_content="",  # Images don't have text content
                            metadata={
                                "bbox": block["bbox"],
                                "image_type": image_ext,
                                "width": block["width"],
                                "height": block["height"],
                                "nearby_text": nearby_text,
                                "type": "image",
                                "page_number": page_num
                            }
                        ))
                except Exception as e:
                    print(f"Error processing image block: {str(e)}")
                    continue

        # Handle tables
        tables = page.find_tables()
        if tables and tables.tables:
            for table in tables.tables:
                df = pd.DataFrame(table.extract())
                documents.append(Document(
                    page_content=df.to_string(),
                    metadata={
                        "bbox": table.bbox,
                        "type": "table",
                        "page_number": page_num
                    }
                ))

    doc.close()
    return documents


def _is_nearby(bbox1: Tuple[float, float, float, float],
               bbox2: Tuple[float, float, float, float],
               threshold: float = 50) -> bool:
    """Check if two bounding boxes are near each other."""
    x1_center = (bbox1[0] + bbox1[2]) / 2
    y1_center = (bbox1[1] + bbox1[3]) / 2
    x2_center = (bbox2[0] + bbox2[2]) / 2
    y2_center = (bbox2[1] + bbox2[3]) / 2

    return (abs(x1_center - x2_center) < threshold and
            abs(y1_center - y2_center) < threshold)


def _rectangles_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    return not (bbox1[2] < bbox2[0] or  # rect1 is left of rect2
                bbox1[0] > bbox2[2] or  # rect1 is right of rect2
                bbox1[3] < bbox2[1] or  # rect1 is above rect2
                bbox1[1] > bbox2[3])  # rect1 is below rect2


def _is_emphasized_text(spans: List[dict]) -> bool:
    """
    Check if text is emphasized (uppercase, bold, or large font).
    """
    # Check various font properties that indicate emphasis
    for span in spans:
        font_properties = span.get("font", "").lower()
        is_bold = any(bold in font_properties for bold in ["bold", "heavy", "black"])
        is_large = span.get("size", 0) > 12
        is_caps = span["text"].isupper()

        if (is_bold and is_caps) or is_large:
            return True
    return False


def _is_unwanted_text(text: str) -> bool:
    """
    Check if text should be filtered out (e.g., copyright notices,
    page numbers, corrupted text with special characters).
    """
    text_lower = text.strip().lower()

    # Patterns to filter out
    unwanted_patterns = [
        "copyright",
        "all rights reserved",
        "©",
        "illustrated encyclopedia",
        "of birds",
        "page",
        "www.",
        ".com",
        "printed in",
        "published by",
    ]

    # Regular expression to detect excessive non-alphanumeric characters
    # Allow common punctuation: ., -, &, etc.
    special_char_pattern = re.compile(r'[^a-zA-Z0-9\s.,&\-]')

    # Check for unwanted patterns
    if any(pattern in text_lower for pattern in unwanted_patterns):
        return True

    # Remove non-printable characters
    clean_text = re.sub(r'[^\x20-\x7E]+', '', text)

    # Check for excessive special characters
    special_chars = special_char_pattern.findall(clean_text)
    if len(special_chars) > len(clean_text) * 0.1:  # more than 10% special chars
        return True

    # Check for very long words without spaces
    if re.search(r'\b\w{30,}\b', clean_text):
        return True

    # Check for high percentage of uppercase letters
    uppercase_letters = re.findall(r'[A-Z]', clean_text)
    if len(uppercase_letters) / len(clean_text) > 0.5:
        return True

    return False


def organize_content(content: Dict[str, dict]) -> Dict[str, dict]:
    """
    Organize extracted content into logical sections.
    Groups related text, tables and images.
    """
    sections = {}
    current_section = None

    for key, item in content.items():
        if item.get("is_title"):
            current_section = item["content"]
            sections[current_section] = {
                "text": [],
                "tables": [],
                "images": []
            }
        elif current_section:
            if key.startswith("text"):
                sections[current_section]["text"].append(item["content"])
            elif key.startswith("table"):
                sections[current_section]["tables"].append(item["content"])
            elif key.startswith("image"):
                sections[current_section]["images"].append(item)

    return sections