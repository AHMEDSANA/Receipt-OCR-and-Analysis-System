# filepath: d:\Hexlar\Reciept File\Paddle_OCR\batch_processor.py
import os
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import zipfile
import io
import cv2
import numpy as np
import re

from receipt_extractor import ReceiptExtractor, ReceiptInfo
from receipt_analyzer import ReceiptAnalyzer


class SimpleImageTextReader:
    """Simplified image text reader for batch processing to avoid circular imports."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the image text reader with optional debug mode."""
        self.debug_mode = debug_mode
        self.debug_images = {}
        self.ocr = None
        
    def _initialize_ocr(self):
        """Initialize PaddleOCR."""
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
        return self.ocr
    
    def process_image(self, image_path: str, preserve_exact_spacing: bool = False) -> Dict[str, Any]:
        """
        Process an image and extract text with preserved layout.
        
        Args:
            image_path: Path to the image
            preserve_exact_spacing: If True, preserve exact spacing without cleaning
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Extract text from the image
            text = self.extract_text_with_layout(image_path)
            
            # Clean the text while preserving layout (unless exact spacing is requested)
            if preserve_exact_spacing:
                cleaned_text = text  # Keep original spacing intact
            else:
                cleaned_text = self._clean_text_preserve_layout(text)
            
            # Create result dictionary
            result = {
                "text": cleaned_text,
                "tables": [],  # Simplified - no table detection in batch mode
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "source_image": os.path.basename(image_path),
                    "ocr_engine": "PaddleOCR",
                    "num_tables_detected": 0
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "source_image": os.path.basename(image_path)
                }
            }
    
    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text from an image using PaddleOCR while preserving layout."""
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
        
        # Sort blocks by y-coordinate (top to bottom)
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
        
        # Construct the final text with preserved spacing
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
                    
                    # Estimate character width
                    char_width = prev_block['width'] / max(len(prev_block['text']), 1)
                    
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
            line = re.sub(r' {6,}', '     ', line)  # Replace 6+ spaces with exactly 5
            
            # Remove other types of whitespace characters but preserve regular spaces
            line = re.sub(r'[^\S ]+', ' ', line)  # Replace tabs, etc. with single space
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class BatchReceiptProcessor:
    """Batch processing utility for multiple receipt images."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the batch processor."""
        self.debug_mode = debug_mode
        self.results = []
        self.errors = []
        
    def process_images_folder(self, folder_path: str, nlp_model=None) -> Dict[str, Any]:
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            nlp_model: spaCy NLP model for text processing
            
        Returns:
            Dictionary with processing results
        """
        # Initialize processors
        image_reader = ImageTextReader(debug_mode=self.debug_mode)
        receipt_extractor = ReceiptExtractor(nlp_model)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f"*{ext}"))
            image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            return {
                'status': 'error',
                'message': 'No image files found in the specified folder',
                'processed_count': 0,
                'results': []
            }
        
        # Initialize processors
        image_reader = SimpleImageTextReader(debug_mode=self.debug_mode)
        receipt_extractor = ReceiptExtractor(nlp_model)
        
        # Process each image
        processed_results = []
        processing_errors = []
        
        progress_bar = st.progress(0) if 'streamlit' in str(type(st)) else None
        status_text = st.empty() if 'streamlit' in str(type(st)) else None
        
        for i, image_path in enumerate(image_files):
            if progress_bar:
                progress = (i + 1) / len(image_files)
                progress_bar.progress(progress)
            
            if status_text:
                status_text.text(f"Processing {image_path.name} ({i+1}/{len(image_files)})")
            
            try:
                # Process the image
                result = self._process_single_image(
                    str(image_path), 
                    image_reader, 
                    receipt_extractor
                )
                
                processed_results.append(result)
                
            except Exception as e:
                error_info = {
                    'file': str(image_path),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                processing_errors.append(error_info)
        
        if status_text:
            status_text.text("✅ Batch processing complete!")
        
        return {
            'status': 'success',
            'processed_count': len(processed_results),
            'error_count': len(processing_errors),
            'results': processed_results,
            'errors': processing_errors,
            'summary': self._generate_batch_summary(processed_results)
        }
    
    def process_uploaded_files(self, uploaded_files: List, nlp_model=None) -> Dict[str, Any]:
        """
        Process multiple uploaded files in Streamlit.
        
        Args:
            uploaded_files: List of uploaded file objects
            nlp_model: spaCy NLP model for text processing
            
        Returns:
            Dictionary with processing results
        """
        # Initialize processors
        image_reader = SimpleImageTextReader(debug_mode=self.debug_mode)
        receipt_extractor = ReceiptExtractor(nlp_model)
        
        processed_results = []
        processing_errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Process the image
                result = self._process_single_image(
                    temp_path, 
                    image_reader, 
                    receipt_extractor,
                    original_filename=uploaded_file.name
                )
                
                processed_results.append(result)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            except Exception as e:
                error_info = {
                    'file': uploaded_file.name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                processing_errors.append(error_info)
        
        status_text.text("✅ Batch processing complete!")
        
        return {
            'status': 'success',
            'processed_count': len(processed_results),
            'error_count': len(processing_errors),
            'results': processed_results,
            'errors': processing_errors,
            'summary': self._generate_batch_summary(processed_results)
        }
    
    def _process_single_image(self, image_path: str, image_reader: SimpleImageTextReader, 
                            receipt_extractor: ReceiptExtractor, 
                            original_filename: str = None) -> Dict[str, Any]:
        """Process a single image and extract receipt information."""
        start_time = datetime.now()
        
        # Extract text from image
        ocr_result = image_reader.process_image(image_path, preserve_exact_spacing=True)
        
        # Extract receipt information
        receipt_info = None
        if ocr_result.get("text"):
            receipt_info = receipt_extractor.extract_receipt_info(ocr_result["text"])
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'filename': original_filename or os.path.basename(image_path),
            'processing_time_seconds': processing_time,
            'ocr_result': ocr_result,
            'receipt_info': receipt_info.__dict__ if receipt_info else None,
            'processed_at': datetime.now().isoformat()
        }
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing."""
        if not results:
            return {}
        
        # Basic stats
        total_files = len(results)
        successful_extractions = sum(1 for r in results if r.get('receipt_info'))
        avg_processing_time = sum(r.get('processing_time_seconds', 0) for r in results) / total_files
        
        # Receipt stats
        total_amount = 0
        merchants = set()
        receipt_types = set()
        items_count = 0
        
        for result in results:
            receipt_info = result.get('receipt_info')
            if receipt_info:
                if receipt_info.get('total'):
                    total_amount += receipt_info['total']
                if receipt_info.get('merchant_name'):
                    merchants.add(receipt_info['merchant_name'])
                if receipt_info.get('receipt_type'):
                    receipt_types.add(receipt_info['receipt_type'])
                if receipt_info.get('items'):
                    items_count += len(receipt_info['items'])
        
        return {
            'total_files_processed': total_files,
            'successful_extractions': successful_extractions,
            'extraction_success_rate': successful_extractions / total_files if total_files > 0 else 0,
            'average_processing_time': avg_processing_time,
            'total_amount_found': total_amount,
            'unique_merchants': len(merchants),
            'unique_receipt_types': len(receipt_types),
            'total_items_extracted': items_count,
            'merchants_list': list(merchants),
            'receipt_types_list': list(receipt_types)
        }
    
    def export_batch_results(self, results: Dict[str, Any], output_format: str = 'json') -> str:
        """
        Export batch processing results to file.
        
        Args:
            results: Results from batch processing
            output_format: 'json', 'csv', or 'excel'
            
        Returns:
            Path to the exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format.lower() == 'json':
            filename = f"batch_results_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                
        elif output_format.lower() == 'csv':
            filename = f"batch_results_{timestamp}.csv"
            self._export_to_csv(results, filename)
            
        elif output_format.lower() == 'excel':
            filename = f"batch_results_{timestamp}.xlsx"
            self._export_to_excel(results, filename)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return filename
    
    def _export_to_csv(self, results: Dict[str, Any], filename: str):
        """Export results to CSV format."""
        rows = []
        
        for result in results.get('results', []):
            receipt_info = result.get('receipt_info', {})
            
            row = {
                'filename': result.get('filename', ''),
                'processing_time': result.get('processing_time_seconds', 0),
                'merchant_name': receipt_info.get('merchant_name', ''),
                'transaction_date': receipt_info.get('transaction_date', ''),
                'transaction_time': receipt_info.get('transaction_time', ''),
                'total': receipt_info.get('total', 0),
                'subtotal': receipt_info.get('subtotal', 0),
                'tax_amount': receipt_info.get('tax_amount', 0),
                'receipt_type': receipt_info.get('receipt_type', ''),
                'payment_method': receipt_info.get('payment_method', ''),
                'confidence_score': receipt_info.get('confidence_score', 0),
                'items_count': len(receipt_info.get('items', [])),
                'processed_at': result.get('processed_at', '')
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    
    def _export_to_excel(self, results: Dict[str, Any], filename: str):
        """Export results to Excel format with multiple sheets."""
        with pd.ExcelWriter(filename) as writer:
            # Summary sheet
            summary_data = []
            if 'summary' in results:
                for key, value in results['summary'].items():
                    summary_data.append({'Metric': key, 'Value': value})
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Receipts sheet
            receipts_data = []
            for result in results.get('results', []):
                receipt_info = result.get('receipt_info', {})
                
                receipts_data.append({
                    'filename': result.get('filename', ''),
                    'processing_time': result.get('processing_time_seconds', 0),
                    'merchant_name': receipt_info.get('merchant_name', ''),
                    'transaction_date': receipt_info.get('transaction_date', ''),
                    'total': receipt_info.get('total', 0),
                    'subtotal': receipt_info.get('subtotal', 0),
                    'tax_amount': receipt_info.get('tax_amount', 0),
                    'receipt_type': receipt_info.get('receipt_type', ''),
                    'confidence_score': receipt_info.get('confidence_score', 0),
                    'items_count': len(receipt_info.get('items', []))
                })
            
            receipts_df = pd.DataFrame(receipts_data)
            receipts_df.to_excel(writer, sheet_name='Receipts', index=False)
            
            # Items sheet
            items_data = []
            for result in results.get('results', []):
                receipt_info = result.get('receipt_info', {})
                filename = result.get('filename', '')
                
                for item in receipt_info.get('items', []):
                    items_data.append({
                        'filename': filename,
                        'merchant_name': receipt_info.get('merchant_name', ''),
                        'transaction_date': receipt_info.get('transaction_date', ''),
                        'item_name': item.get('name', ''),
                        'price': item.get('price', 0),
                        'quantity': item.get('quantity', 1),
                        'category': item.get('category', '')
                    })
            
            if items_data:
                items_df = pd.DataFrame(items_data)
                items_df.to_excel(writer, sheet_name='Items', index=False)
            
            # Errors sheet
            if results.get('errors'):
                errors_df = pd.DataFrame(results['errors'])
                errors_df.to_excel(writer, sheet_name='Errors', index=False)
    
    def create_batch_processing_ui(self):
        """Create Streamlit UI for batch processing."""
        st.header("📦 Batch Receipt Processing")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Upload Multiple Files", "Process Folder"],
                help="Choose how to provide receipt images"
            )
        
        with col2:
            extract_receipt_info = st.checkbox(
                "Extract Receipt Information",
                value=True,
                help="Extract structured data from receipts"
            )
        
        if processing_mode == "Upload Multiple Files":
            st.subheader("📁 Upload Receipt Images")
            uploaded_files = st.file_uploader(
                "Choose receipt image files",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple receipt images for batch processing"
            )
            
            if uploaded_files:
                st.info(f"Selected {len(uploaded_files)} files for processing")
                
                # Show file list
                with st.expander("📋 File List"):
                    for i, file in enumerate(uploaded_files, 1):
                        st.text(f"{i}. {file.name} ({file.size} bytes)")
                
                if st.button("🚀 Process All Files", type="primary"):
                    with st.spinner("Processing images..."):
                        # Load spaCy model directly to avoid circular imports
                        import spacy
                        try:
                            nlp = spacy.load("en_core_web_sm")
                        except OSError:
                            import subprocess
                            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                            nlp = spacy.load("en_core_web_sm")
                        
                        results = self.process_uploaded_files(uploaded_files, nlp)
                        
                        # Display results
                        self._display_batch_results(results)
        
        else:  # Process Folder
            st.subheader("📂 Process Folder")
            folder_path = st.text_input(
                "Folder Path",
                placeholder="Enter path to folder containing receipt images",
                help="Provide the full path to the folder containing receipt images"
            )
            
            if folder_path and st.button("🚀 Process Folder", type="primary"):
                if os.path.exists(folder_path):
                    with st.spinner("Processing images..."):
                        # Load spaCy model directly to avoid circular imports
                        import spacy
                        try:
                            nlp = spacy.load("en_core_web_sm")
                        except OSError:
                            import subprocess
                            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                            nlp = spacy.load("en_core_web_sm")
                        
                        results = self.process_images_folder(folder_path, nlp)
                        
                        # Display results
                        self._display_batch_results(results)
                else:
                    st.error("Folder path does not exist. Please check the path and try again.")
    
    def _display_batch_results(self, results: Dict[str, Any]):
        """Display batch processing results in Streamlit."""
        if results['status'] == 'error':
            st.error(f"Processing failed: {results['message']}")
            return
        
        # Summary metrics
        st.subheader("📊 Processing Summary")
        summary = results.get('summary', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Processed", results['processed_count'])
        with col2:
            st.metric("Success Rate", f"{summary.get('extraction_success_rate', 0):.1%}")
        with col3:
            st.metric("Total Amount", f"${summary.get('total_amount_found', 0):.2f}")
        with col4:
            st.metric("Unique Merchants", summary.get('unique_merchants', 0))
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Items Extracted", summary.get('total_items_extracted', 0))
        with col2:
            st.metric("Avg Processing Time", f"{summary.get('average_processing_time', 0):.2f}s")
        with col3:
            st.metric("Errors", results['error_count'])
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Results Overview", "🧾 Individual Receipts", "❌ Errors", "💾 Export"])
        
        with tab1:
            st.subheader("Merchants Found")
            if summary.get('merchants_list'):
                for merchant in summary['merchants_list']:
                    st.text(f"• {merchant}")
            else:
                st.info("No merchants identified")
            
            st.subheader("Receipt Types")
            if summary.get('receipt_types_list'):
                for receipt_type in summary['receipt_types_list']:
                    st.text(f"• {receipt_type.title()}")
            else:
                st.info("No receipt types identified")
        
        with tab2:
            st.subheader("Individual Receipt Details")
            
            for i, result in enumerate(results['results'], 1):
                receipt_info = result.get('receipt_info')
                
                with st.expander(f"Receipt {i}: {result['filename']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text(f"Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
                        
                        if receipt_info:
                            st.text(f"Merchant: {receipt_info.get('merchant_name', 'N/A')}")
                            st.text(f"Date: {receipt_info.get('transaction_date', 'N/A')}")
                            st.text(f"Total: ${receipt_info.get('total', 0):.2f}")
                            st.text(f"Type: {receipt_info.get('receipt_type', 'N/A')}")
                    
                    with col2:
                        if receipt_info:
                            st.text(f"Items: {len(receipt_info.get('items', []))}")
                            st.text(f"Confidence: {receipt_info.get('confidence_score', 0):.1%}")
                            st.text(f"Payment: {receipt_info.get('payment_method', 'N/A')}")
        
        with tab3:
            if results['errors']:
                st.subheader("Processing Errors")
                for error in results['errors']:
                    st.error(f"**{error['file']}**: {error['error']}")
            else:
                st.success("No errors occurred during processing!")
        
        with tab4:
            st.subheader("Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export JSON"):
                    filename = self.export_batch_results(results, 'json')
                    st.success(f"Exported to {filename}")
            
            with col2:
                if st.button("📊 Export CSV"):
                    filename = self.export_batch_results(results, 'csv')
                    st.success(f"Exported to {filename}")
            
            with col3:
                if st.button("📋 Export Excel"):
                    filename = self.export_batch_results(results, 'excel')
                    st.success(f"Exported to {filename}")
            
            # Download buttons for the exported files
            st.markdown("---")
            st.subheader("Download Files")
            
            # Create downloadable files
            json_data = json.dumps(results, indent=2, default=str)
            csv_data = self._create_csv_download_data(results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def _create_csv_download_data(self, results: Dict[str, Any]) -> str:
        """Create CSV data for download."""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        header = [
            'filename', 'processing_time', 'merchant_name', 'transaction_date',
            'total', 'subtotal', 'tax_amount', 'receipt_type', 'payment_method',
            'confidence_score', 'items_count'
        ]
        writer.writerow(header)
        
        # Write data
        for result in results.get('results', []):
            receipt_info = result.get('receipt_info', {})
            
            row = [
                result.get('filename', ''),
                result.get('processing_time_seconds', 0),
                receipt_info.get('merchant_name', ''),
                receipt_info.get('transaction_date', ''),
                receipt_info.get('total', 0),
                receipt_info.get('subtotal', 0),
                receipt_info.get('tax_amount', 0),
                receipt_info.get('receipt_type', ''),
                receipt_info.get('payment_method', ''),
                receipt_info.get('confidence_score', 0),
                len(receipt_info.get('items', []))
            ]
            writer.writerow(row)
        
        return output.getvalue()