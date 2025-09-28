import cv2
import numpy as np
import re
import spacy
import json
import os
import streamlit as st
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import zipfile
import io
from receipt_extractor import ReceiptExtractor, ReceiptInfo
from batch_processor import BatchReceiptProcessor
from receipt_analyzer import ReceiptAnalyzer

# Load spaCy NLP model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Auto-download if not available
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

class ImageTextReader:
    def __init__(self, debug_mode: bool = False):
        """Initialize the image text reader with optional debug mode."""
        self.debug_mode = debug_mode
        self.debug_images = {}

        # Initialize PaddleOCR
        self._initialize_ocr()

    @st.cache_resource
    def _initialize_ocr(_self):
        """Initialize PaddleOCR with caching."""
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Apply advanced preprocessing techniques to improve OCR accuracy.

        Args:
            image_path: Path to the image

        Returns:
            Preprocessed image as numpy array
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Store original for debugging
        if self.debug_mode:
            self.debug_images["original"] = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug_mode:
            self.debug_images["grayscale"] = gray.copy()

        # Apply deskewing if needed
        angle = self._get_skew_angle(gray)
        if abs(angle) > 0.5:
            gray = self._deskew(gray, angle)
            if self.debug_mode:
                self.debug_images["deskewed"] = gray.copy()

        # Noise removal with bilateral filter (preserves edges better than Gaussian)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        if self.debug_mode:
            self.debug_images["denoised"] = denoised.copy()

        # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        if self.debug_mode:
            self.debug_images["enhanced"] = enhanced.copy()

        # Adaptive thresholding
        thresholded = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        if self.debug_mode:
            self.debug_images["thresholded"] = thresholded.copy()

        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        if self.debug_mode:
            self.debug_images["cleaned"] = cleaned.copy()

        return cleaned

    def _get_skew_angle(self, image: np.ndarray) -> float:
        """Detect the skew angle of the image."""
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is None:
            return 0.0

        # Calculate the average angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            if theta < np.pi/4 or theta > 3*np.pi/4:  # Only consider near-vertical lines
                angles.append(theta)

        if not angles:
            return 0.0

        # Convert to degrees and normalize
        avg_angle = np.mean(angles) * 180 / np.pi
        if avg_angle > 45:
            avg_angle -= 90

        return avg_angle

    def _deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image to correct skew."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _estimate_character_width(self, text: str, text_width: float) -> float:
        """
        Estimate average character width based on text content and bounding box width.
        
        Args:
            text: The text string
            text_width: Width of the text bounding box
            
        Returns:
            Estimated average character width in pixels
        """
        if not text or len(text) == 0:
            return 10.0  # Default character width
        
        # Account for different character widths (rough estimation)
        char_count = len(text)
        
        # Adjust for common wide characters
        wide_chars = sum(1 for c in text if c in 'MWQO@#%&')
        narrow_chars = sum(1 for c in text if c in 'iltj|!()[]{}.,;:')
        
        # Weighted character count
        adjusted_count = char_count + (wide_chars * 0.3) - (narrow_chars * 0.3)
        
        return max(text_width / max(adjusted_count, 1), 3.0)  # Minimum 3px per character

    def extract_text_with_layout(self, image_path: str) -> str:
        """
        Extract text from an image using PaddleOCR while preserving the original layout.

        Args:
            image_path: Path to the image

        Returns:
            Extracted text as a string with original layout preserved
        """
        # Get OCR instance
        ocr = self._initialize_ocr()
        
        # Process the image with PaddleOCR
        result = ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return ""

        # Get image dimensions
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        # Extract text with position information
        text_blocks = []

        for line in result[0]:
            if len(line) >= 2:  # Ensure we have both coordinates and text
                box = line[0]  # Get the bounding box coordinates
                text_info = line[1]  # Get the text and confidence
                
                if isinstance(text_info, tuple) and len(text_info) >= 1:
                    text = text_info[0]  # Extract the text
                    confidence = text_info[1] if len(text_info) > 1 else 0  # Extract confidence
                    
                    # Calculate center y-coordinate of the text box
                    y_center = sum(point[1] for point in box) / 4
                    
                    # Calculate left and right x-coordinates for better spacing
                    x_left = min(point[0] for point in box)
                    x_right = max(point[0] for point in box)
                    
                    text_blocks.append({
                        'text': text,
                        'y_center': y_center,
                        'x_left': x_left,
                        'x_right': x_right,
                        'width': x_right - x_left,
                        'confidence': confidence,
                        'box': box
                    })
                    
                    if self.debug_mode:
                        st.write(f"Text: {text}, Position: ({x_left:.1f}, {y_center:.1f}), Confidence: {confidence:.2f}")

        # Sort blocks by y-coordinate (top to bottom)
        # Use a threshold to group text blocks that are on the same line
        line_threshold = img_height * 0.02  # 2% of image height
        
        # Group text blocks into lines
        lines = []
        current_line = []
        
        # Sort initially by y_center to process from top to bottom
        sorted_blocks = sorted(text_blocks, key=lambda b: b['y_center'])
        
        if sorted_blocks:
            current_line = [sorted_blocks[0]]
            last_y = sorted_blocks[0]['y_center']
            
            for block in sorted_blocks[1:]:
                # If this block is far enough from the last one, it's a new line
                if abs(block['y_center'] - last_y) > line_threshold:
                    # Sort the current line by x_left before adding to lines
                    current_line.sort(key=lambda b: b['x_left'])
                    lines.append(current_line)
                    current_line = [block]
                    last_y = block['y_center']
                else:
                    current_line.append(block)
            
            # Don't forget the last line
            if current_line:
                current_line.sort(key=lambda b: b['x_left'])
                lines.append(current_line)
        
        # Construct the final text with preserved spacing based on positions
        result_text = ""
        for line in lines:
            if not line:
                continue
                
            line_text = ""
            for i, block in enumerate(line):
                if i == 0:
                    line_text += block['text']
                else:
                    # Calculate spacing between consecutive blocks
                    prev_block = line[i-1]
                    gap = block['x_left'] - prev_block['x_right']
                    
                    # Estimate character width using improved method
                    char_width = self._estimate_character_width(prev_block['text'], prev_block['width'])
                    
                    # Calculate number of spaces based on gap
                    if gap > char_width * 0.5:  # Meaningful gap detected
                        num_spaces = max(1, int(gap / char_width))
                        # Limit excessive spacing while preserving intentional gaps
                        num_spaces = min(num_spaces, 25)  # Max 25 spaces
                        line_text += " " * num_spaces + block['text']
                    else:
                        # If blocks are close together, use single space
                        line_text += " " + block['text']
            
            result_text += line_text + "\n"
        
        return result_text.rstrip()  # Remove trailing newline but preserve internal structure

    def detect_tables(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect tables in the image and extract their structure.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of detected tables with their structure and content
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to enhance table structure
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        if self.debug_mode:
            self.debug_images["table_binary"] = binary.copy()
            self.debug_images["table_dilated"] = dilated.copy()
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find potential tables
        tables = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Minimum area threshold for tables
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio to filter out non-table regions
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5:  # Tables typically have reasonable aspect ratios
                    table_region = {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area
                    }
                    tables.append(table_region)
                    
                    if self.debug_mode:
                        # Draw rectangle around detected table
                        table_img = image.copy()
                        cv2.rectangle(table_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        self.debug_images[f"table_detected_{len(tables)}"] = table_img
        
        # Process each detected table
        processed_tables = []
        for i, table in enumerate(tables):
            # Extract the table region
            table_img = image[table['y']:table['y']+table['height'], 
                             table['x']:table['x']+table['width']]
            
            # Save table region for OCR
            temp_path = f"temp_table_{i}.jpg"
            cv2.imwrite(temp_path, table_img)
            
            # Extract text from the table region
            table_text = self.extract_table_content(temp_path, table)
            
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            processed_tables.append(table_text)
        
        return processed_tables

    def extract_table_content(self, table_image_path: str, table_region: Dict[str, int]) -> Dict[str, Any]:
        """
        Extract content from a table image and organize it into a structured format.
        
        Args:
            table_image_path: Path to the table image
            table_region: Dictionary with table region coordinates
            
        Returns:
            Dictionary with table structure and content
        """
        # Get OCR instance
        ocr = self._initialize_ocr()
        
        # Process the table image with PaddleOCR
        result = ocr.ocr(table_image_path, cls=True)
        
        if not result or not result[0]:
            return {
                "region": table_region,
                "cells": [],
                "raw_text": "",
                "formatted_text": ""
            }
        
        # Extract text blocks with position information
        text_blocks = []
        for line in result[0]:
            if len(line) >= 2:
                box = line[0]
                text_info = line[1]
                
                if isinstance(text_info, tuple) and len(text_info) >= 1:
                    text = text_info[0]
                    confidence = text_info[1] if len(text_info) > 1 else 0
                    
                    # Calculate center coordinates
                    y_center = sum(point[1] for point in box) / 4
                    x_center = sum(point[0] for point in box) / 4
                    
                    # Calculate bounding box dimensions
                    x_min = min(point[0] for point in box)
                    y_min = min(point[1] for point in box)
                    x_max = max(point[0] for point in box)
                    y_max = max(point[1] for point in box)
                    
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'x_center': x_center,
                        'y_center': y_center,
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                        'box': box
                    })
        
        # Detect table structure (rows and columns)
        # First, identify potential row positions
        if not text_blocks:
            return {
                "region": table_region,
                "cells": [],
                "raw_text": "",
                "formatted_text": ""
            }
            
        # Sort blocks by y-center to find rows
        sorted_by_y = sorted(text_blocks, key=lambda b: b['y_center'])
        
        # Get image dimensions
        img = cv2.imread(table_image_path)
        img_height, img_width = img.shape[:2]
        
        # Use a threshold to group text blocks into rows
        row_threshold = img_height * 0.03  # 3% of image height
        
        # Group into rows
        rows = []
        current_row = [sorted_by_y[0]]
        last_y = sorted_by_y[0]['y_center']
        
        for block in sorted_by_y[1:]:
            if abs(block['y_center'] - last_y) > row_threshold:
                rows.append(current_row)
                current_row = [block]
                last_y = block['y_center']
            else:
                current_row.append(block)
                # Update last_y to be the average of the row
                last_y = sum(b['y_center'] for b in current_row) / len(current_row)
        
        # Add the last row
        if current_row:
            rows.append(current_row)
        
        # Now identify columns
        # First, find all potential column positions by sorting blocks by x-center
        all_blocks = [block for row in rows for block in row]
        sorted_by_x = sorted(all_blocks, key=lambda b: b['x_center'])
        
        # Use a threshold to identify potential column boundaries
        column_threshold = img_width * 0.05  # 5% of image width
        
        # Find potential column centers
        column_centers = []
        if sorted_by_x:
            current_center = sorted_by_x[0]['x_center']
            column_centers.append(current_center)
            
            for block in sorted_by_x[1:]:
                if abs(block['x_center'] - current_center) > column_threshold:
                    current_center = block['x_center']
                    column_centers.append(current_center)
        
        # Sort column centers from left to right
        column_centers.sort()
        
        # Assign each text block to a cell (row, column)
        cells = []
        for row_idx, row in enumerate(rows):
            for block in row:
                # Find the closest column center
                col_idx = min(range(len(column_centers)), 
                             key=lambda i: abs(column_centers[i] - block['x_center']))
                
                cells.append({
                    'row': row_idx,
                    'column': col_idx,
                    'text': block['text'],
                    'confidence': block['confidence'],
                    'position': {
                        'x_center': block['x_center'],
                        'y_center': block['y_center'],
                        'x_min': block['x_min'],
                        'y_min': block['y_min'],
                        'x_max': block['x_max'],
                        'y_max': block['y_max']
                    }
                })
        
        # Create a formatted text representation of the table
        formatted_text = self._format_table(cells, len(rows), len(column_centers))
        
        # Create raw text by preserving spacing between text blocks
        # Sort all text blocks by position for better spacing calculation
        sorted_text_blocks = sorted(text_blocks, key=lambda b: (b['y_center'], b['x_center']))
        
        raw_text = ""
        for i, block in enumerate(sorted_text_blocks):
            if i == 0:
                raw_text = block['text']
            else:
                prev_block = sorted_text_blocks[i-1]
                # Calculate if blocks are on the same line
                same_line = abs(block['y_center'] - prev_block['y_center']) < (img_height * 0.03)
                
                if same_line:
                    # Calculate spacing for same line
                    gap = block['x_min'] - prev_block['x_max']
                    if gap > 20:  # Significant gap, add multiple spaces
                        spaces = max(2, min(int(gap / 10), 8))  # 2-8 spaces based on gap
                        raw_text += " " * spaces + block['text']
                    else:
                        raw_text += " " + block['text']
                else:
                    # Different line, add newline
                    raw_text += "\n" + block['text']
        
        return {
            "region": table_region,
            "cells": cells,
            "num_rows": len(rows),
            "num_columns": len(column_centers),
            "raw_text": raw_text,
            "formatted_text": formatted_text
        }
    
    def _format_table(self, cells: List[Dict[str, Any]], num_rows: int, num_columns: int) -> str:
        """
        Format table cells into a readable text representation.
        
        Args:
            cells: List of cell dictionaries
            num_rows: Number of rows in the table
            num_columns: Number of columns in the table
            
        Returns:
            Formatted table as text
        """
        # Create an empty grid
        grid = [['' for _ in range(num_columns)] for _ in range(num_rows)]
        
        # Fill in the grid with cell text
        for cell in cells:
            row = cell['row']
            col = cell['column']
            if 0 <= row < num_rows and 0 <= col < num_columns:
                if grid[row][col]:
                    grid[row][col] += " " + cell['text']
                else:
                    grid[row][col] = cell['text']
        
        # Find the maximum width needed for each column
        col_widths = [0] * num_columns
        for row in grid:
            for col_idx, cell_text in enumerate(row):
                col_widths[col_idx] = max(col_widths[col_idx], len(cell_text) + 2)  # +2 for padding
        
        # Create the formatted table
        formatted_table = ""
        
        # Add top border
        border = "+"
        for width in col_widths:
            border += "-" * width + "+"
        formatted_table += border + "\n"
        
        # Add rows
        for row in grid:
            row_text = "|"
            for col_idx, cell_text in enumerate(row):
                padding = col_widths[col_idx] - len(cell_text)
                left_padding = padding // 2
                right_padding = padding - left_padding
                row_text += " " * left_padding + cell_text + " " * right_padding + "|"
            formatted_table += row_text + "\n"
            
            # Add separator after each row
            formatted_table += border + "\n"
        
        return formatted_table

    def _clean_text_preserve_layout(self, text: str) -> str:
        """Clean text while preserving layout and intentional spacing."""
        # Remove non-printable characters except newlines and spaces
        text = ''.join(c for c in text if c.isprintable() or c == '\n')
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        
        # Clean each line individually while preserving spacing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace only
            line = line.strip()
            
            # Only collapse excessive spaces (more than 5 consecutive spaces)
            # This preserves intentional formatting while cleaning up noise
            line = re.sub(r' {6,}', '     ', line)  # Replace 6+ spaces with exactly 5
            
            # Remove other types of whitespace characters but preserve regular spaces
            line = re.sub(r'[^\S ]+', ' ', line)  # Replace tabs, etc. with single space
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def save_debug_images(self, output_dir: str) -> Dict[str, str]:
        """Save debug images to the specified directory and return file paths."""
        if not self.debug_mode or not self.debug_images:
            return {}

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        for name, img in self.debug_images.items():
            output_path = os.path.join(output_dir, f"{name}.jpg")
            cv2.imwrite(output_path, img)
            saved_files[name] = output_path

        return saved_files

    def process_image(self, image_path: str, output_json_path: Optional[str] = None, output_text_path: Optional[str] = None, preserve_exact_spacing: bool = False) -> Dict[str, Any]:
        """
        Process an image and extract text with preserved layout and tables.

        Args:
            image_path: Path to the image
            output_json_path: Optional path to save the JSON output
            output_text_path: Optional path to save the text output
            preserve_exact_spacing: If True, preserve exact spacing without cleaning

        Returns:
            Dictionary containing extracted text, tables, and metadata
        """
        try:
            # Extract text from the image with layout preserved
            text = self.extract_text_with_layout(image_path)
            
            # Clean the text while preserving layout (unless exact spacing is requested)
            if preserve_exact_spacing:
                cleaned_text = text  # Keep original spacing intact
            else:
                cleaned_text = self._clean_text_preserve_layout(text)
            
            # Detect and extract tables
            tables = self.detect_tables(image_path)

            # Create result dictionary
            result = {
                "text": cleaned_text,
                "tables": tables,
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "source_image": os.path.basename(image_path),
                    "ocr_engine": "PaddleOCR",
                    "num_tables_detected": len(tables)
                }
            }

            # Save to JSON if path provided
            if output_json_path:
                with open(output_json_path, "w", encoding="utf-8") as json_file:
                    json.dump(result, json_file, indent=4)
            
            # Save to plain text file if path provided
            if output_text_path:
                with open(output_text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(cleaned_text)
                    
                    # Add tables to the text file
                    if tables:
                        text_file.write("\n\n--- DETECTED TABLES ---\n\n")
                        for i, table in enumerate(tables):
                            text_file.write(f"Table {i+1}:\n")
                            text_file.write(table["formatted_text"])
                            text_file.write("\n\n")

            return result

        except Exception as e:
            error_data = {
                "error": str(e),
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "source_image": os.path.basename(image_path)
                }
            }

            if output_json_path:
                with open(output_json_path, "w", encoding="utf-8") as json_file:
                    json.dump(error_data, json_file, indent=4)

            return error_data


def main():
    st.set_page_config(
        page_title="Image Text Extraction Tool (PaddleOCR)",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📝 Image Text Extraction Tool (PaddleOCR)")
    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Processing mode selection
    processing_mode = st.sidebar.selectbox(
        "Processing Mode",
        ["Single Image", "Batch Processing", "Analysis Dashboard"],
        help="Choose processing mode"
    )
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("Enable debug mode", value=False, help="Generate debug images showing processing steps")
    
    # Table detection
    detect_tables = st.sidebar.checkbox("Detect tables", value=True, help="Attempt to detect and structure tables in the image")
    
    # Spacing preservation option
    preserve_exact_spacing = st.sidebar.checkbox("Preserve exact spacing", value=True, help="Preserve original spacing between text elements without cleaning")
    
    # Receipt extraction
    extract_receipt_info = st.sidebar.checkbox("Extract receipt information", value=True, help="Extract structured receipt data including items, prices, and merchant details")
    
    # File upload
    st.sidebar.markdown("### 📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file for text extraction"
    )

    # Main content based on processing mode
    if processing_mode == "Batch Processing":
        batch_processor = BatchReceiptProcessor(debug_mode=debug_mode)
        batch_processor.create_batch_processing_ui()
    
    elif processing_mode == "Analysis Dashboard":
        analyzer = ReceiptAnalyzer()
        analyzer.create_spending_dashboard()
    
    elif processing_mode == "Single Image" and uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.info(f"**Format:** {image.format}")

        with col2:
            st.subheader("🔄 Processing")
            
            # Process button
            if st.button("🚀 Process Image", type="primary"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_image_path = tmp_file.name

                try:
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Initializing OCR engine...")
                    progress_bar.progress(10)
                    
                    # Initialize reader
                    reader = ImageTextReader(debug_mode=debug_mode)
                    
                    status_text.text("Processing image...")
                    progress_bar.progress(30)
                    
                    # Process the image
                    result = reader.process_image(temp_image_path, preserve_exact_spacing=preserve_exact_spacing)
                    
                    progress_bar.progress(60)
                    
                    # Extract receipt information if enabled
                    receipt_info = None
                    if extract_receipt_info and result.get("text"):
                        status_text.text("Extracting receipt information...")
                        receipt_extractor = ReceiptExtractor(nlp)
                        receipt_info = receipt_extractor.extract_receipt_info(result["text"])
                        result["receipt_info"] = receipt_info
                    
                    progress_bar.progress(80)
                    status_text.text("Generating outputs...")
                    
                    # Prepare output files
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    
                    # Text output
                    text_output = result["text"]
                    if result["tables"]:
                        text_output += "\n\n--- DETECTED TABLES ---\n\n"
                        for i, table in enumerate(result["tables"]):
                            text_output += f"Table {i+1}:\n"
                            text_output += table["formatted_text"]
                            text_output += "\n\n"
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📄 Results")
                    
                    # Create tabs for different views
                    tabs = ["📝 Extracted Text", "📊 Tables", "📋 Metadata", "💾 Downloads"]
                    if receipt_info:
                        tabs.insert(1, "🧾 Receipt Info")
                    
                    if len(tabs) == 4:
                        tab1, tab2, tab3, tab4 = st.tabs(tabs)
                    else:
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)
                    
                    with tab1:
                        st.markdown("### Extracted Text")
                        if preserve_exact_spacing:
                            st.info("📏 **Exact spacing preservation enabled** - Original spacing between text elements is maintained")
                        else:
                            st.info("🧹 **Text cleaning enabled** - Excessive spacing is reduced while preserving layout")
                        
                        if result["text"]:
                            st.text_area("Extracted text with preserved layout:", 
                                       value=result["text"], 
                                       height=400, 
                                       key="extracted_text")
                        else:
                            st.warning("No text was extracted from the image.")
                    
                    # Handle receipt information tab if it exists
                    if receipt_info:
                        with tab2:
                            st.markdown("### Receipt Information")
                            
                            # Display confidence score
                            confidence = receipt_info.confidence_score or 0
                            confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                            st.markdown(f"**Extraction Confidence:** <span style='color: {confidence_color}'>{confidence:.1%}</span>", unsafe_allow_html=True)
                            
                            # Create columns for organized display
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 🏪 Merchant Information")
                                if receipt_info.merchant_name:
                                    st.text(f"Name: {receipt_info.merchant_name}")
                                if receipt_info.merchant_address:
                                    st.text(f"Address: {receipt_info.merchant_address}")
                                if receipt_info.merchant_phone:
                                    st.text(f"Phone: {receipt_info.merchant_phone}")
                                if receipt_info.merchant_email:
                                    st.text(f"Email: {receipt_info.merchant_email}")
                                
                                st.markdown("#### 💰 Financial Summary")
                                if receipt_info.subtotal is not None:
                                    st.text(f"Subtotal: ${receipt_info.subtotal:.2f}")
                                if receipt_info.tax_amount is not None:
                                    st.text(f"Tax: ${receipt_info.tax_amount:.2f}")
                                if receipt_info.discount is not None:
                                    st.text(f"Discount: ${receipt_info.discount:.2f}")
                                if receipt_info.total is not None:
                                    st.text(f"Total: ${receipt_info.total:.2f}")
                            
                            with col2:
                                st.markdown("#### 📋 Transaction Details")
                                if receipt_info.transaction_date:
                                    st.text(f"Date: {receipt_info.transaction_date}")
                                if receipt_info.transaction_time:
                                    st.text(f"Time: {receipt_info.transaction_time}")
                                if receipt_info.receipt_number:
                                    st.text(f"Receipt #: {receipt_info.receipt_number}")
                                if receipt_info.receipt_type:
                                    st.text(f"Type: {receipt_info.receipt_type.title()}")
                                
                                st.markdown("#### 💳 Payment Information")
                                if receipt_info.payment_method:
                                    st.text(f"Method: {receipt_info.payment_method}")
                                if receipt_info.card_last_four:
                                    st.text(f"Card ending in: {receipt_info.card_last_four}")
                                if receipt_info.amount_paid is not None:
                                    st.text(f"Amount Paid: ${receipt_info.amount_paid:.2f}")
                                if receipt_info.change_given is not None:
                                    st.text(f"Change: ${receipt_info.change_given:.2f}")
                            
                            # Items section
                            if receipt_info.items:
                                st.markdown("#### 🛒 Items")
                                items_df_data = []
                                for i, item in enumerate(receipt_info.items, 1):
                                    items_df_data.append({
                                        "#": i,
                                        "Item": item.name,
                                        "Quantity": item.quantity or 1,
                                        "Price": f"${item.price:.2f}",
                                        "Category": item.category or "N/A"
                                    })
                                
                                st.dataframe(items_df_data, use_container_width=True)
                                
                                # Summary stats
                                total_items = len(receipt_info.items)
                                items_total = sum(item.price for item in receipt_info.items)
                                avg_price = items_total / total_items if total_items > 0 else 0
                                
                                col_stats1, col_stats2, col_stats3 = st.columns(3)
                                with col_stats1:
                                    st.metric("Total Items", total_items)
                                with col_stats2:
                                    st.metric("Items Total", f"${items_total:.2f}")
                                with col_stats3:
                                    st.metric("Average Price", f"${avg_price:.2f}")
                            
                            # Receipt summary
                            st.markdown("#### 📄 Receipt Summary")
                            summary = receipt_extractor.format_receipt_summary(receipt_info)
                            st.text_area("Formatted Summary", value=summary, height=300)
                        
                        # Update tab reference for tables
                        tab_tables = tab3
                    else:
                        tab_tables = tab2
                    
                    with tab_tables:
                        st.markdown("### Detected Tables")
                        if result["tables"]:
                            st.success(f"Found {len(result['tables'])} table(s)")
                            
                            for i, table in enumerate(result["tables"]):
                                with st.expander(f"Table {i+1}", expanded=True):
                                    st.markdown("**Table Structure:**")
                                    st.code(table["formatted_text"], language="text")
                                    
                                    st.markdown("**Raw Text:**")
                                    st.text(table["raw_text"])
                                    
                                    if "num_rows" in table and "num_columns" in table:
                                        st.info(f"Dimensions: {table['num_rows']} rows × {table['num_columns']} columns")
                        else:
                            st.info("No tables were detected in the image.")
                    
                    # Handle metadata and downloads tabs with proper indexing
                    if receipt_info:
                        with tab4:
                            st.markdown("### Processing Metadata")
                            metadata = result["metadata"]
                            
                            col_meta1, col_meta2 = st.columns(2)
                            with col_meta1:
                                st.metric("Processing Time", metadata["processed_at"])
                                st.metric("OCR Engine", metadata["ocr_engine"])
                            
                            with col_meta2:
                                st.metric("Source Image", metadata["source_image"])
                                st.metric("Tables Detected", metadata["num_tables_detected"])
                        
                        with tab5:
                            st.markdown("### Download Results")
                    else:
                        with tab3:
                            st.markdown("### Processing Metadata")
                            metadata = result["metadata"]
                            
                            col_meta1, col_meta2 = st.columns(2)
                            with col_meta1:
                                st.metric("Processing Time", metadata["processed_at"])
                                st.metric("OCR Engine", metadata["ocr_engine"])
                            
                            with col_meta2:
                                st.metric("Source Image", metadata["source_image"])
                                st.metric("Tables Detected", metadata["num_tables_detected"])
                        
                        with tab4:
                            st.markdown("### Download Results")
                            
                            col_dl1, col_dl2 = st.columns(2)
                            
                            with col_dl1:
                                # JSON download
                                def serialize_receipt_info(obj):
                                    if hasattr(obj, '__dict__'):
                                        return obj.__dict__
                                    return str(obj)
                                
                                json_output = json.dumps(result, indent=4, default=serialize_receipt_info)
                                
                                st.download_button(
                                    label="📄 Download JSON",
                                    data=json_output,
                                    file_name=f"{base_name}_text.json",
                                    mime="application/json",
                                    key="download_json"
                                )
                            
                            with col_dl2:
                                # Text download
                                st.download_button(
                                    label="📝 Download Text",
                                    data=text_output,
                                    file_name=f"{base_name}_text.txt",
                                    mime="text/plain",
                                    key="download_text"
                                )
                    
                    # Handle download section for both scenarios
                    if not receipt_info:
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # JSON download
                            def serialize_receipt_info(obj):
                                if hasattr(obj, '__dict__'):
                                    return obj.__dict__
                                return str(obj)
                            
                            json_output = json.dumps(result, indent=4, default=serialize_receipt_info)
                            
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_output,
                                file_name=f"{base_name}_text.json",
                                mime="application/json",
                                key="download_json"
                            )
                        
                        with col_dl2:
                            # Text download
                            st.download_button(
                                label="📝 Download Text",
                                data=text_output,
                                file_name=f"{base_name}_text.txt",
                                mime="text/plain",
                                key="download_text"
                            )
                        
                        # Debug images download
                        if debug_mode and reader.debug_images:
                            st.markdown("### Debug Images")
                            
                            # Create ZIP file with debug images
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for name, img in reader.debug_images.items():
                                    # Convert OpenCV image to bytes
                                    is_success, buffer = cv2.imencode(".jpg", img)
                                    if is_success:
                                        zip_file.writestr(f"{name}.jpg", buffer.tobytes())
                            
                            zip_buffer.seek(0)
                            
                            st.download_button(
                                label="🔧 Download Debug Images (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"{base_name}_debug_images.zip",
                                mime="application/zip",
                                key="download_debug"
                            )
                            
                            # Display debug images
                            st.markdown("### Debug Image Preview")
                            debug_cols = st.columns(3)
                            
                            for i, (name, img) in enumerate(reader.debug_images.items()):
                                col_idx = i % 3
                                with debug_cols[col_idx]:
                                    # Convert BGR to RGB for display
                                    if len(img.shape) == 3:
                                        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    else:
                                        display_img = img
                                    
                                    st.image(display_img, caption=name.replace('_', ' ').title(), use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred while processing the image: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)

    else:
        # Instructions when no file is uploaded
        st.markdown("""
        ### 📋 Instructions
        
        1. **Upload an image** using the file uploader in the sidebar
        2. **Configure options** in the sidebar:
           - Enable debug mode to see image processing steps
           - Enable table detection to extract structured data
        3. **Click "Process Image"** to extract text and tables
        4. **View results** in the tabs below
        5. **Download** the extracted data in JSON or text format
        
        ### 🎯 Supported Features
        
        - **Text Extraction**: Extract text while preserving original layout
        - **Table Detection**: Automatically detect and structure tables
        - **Multiple Formats**: Support for PNG, JPG, JPEG, BMP, TIFF
        - **Debug Mode**: View intermediate processing steps
        - **Export Options**: Download results as JSON or plain text
        
        ### 💡 Tips for Better Results
        
        - Use high-resolution images for better accuracy
        - Ensure good contrast between text and background
        - Avoid skewed or rotated images (auto-correction is limited)
        - For tables, ensure clear borders and good alignment
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by PaddleOCR • Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
