# Receipt OCR and Analysis System

A comprehensive receipt processing system built with PaddleOCR, featuring intelligent text extraction, receipt data parsing, batch processing, and spending analysis capabilities.

## 🚀 Features

### Core Functionality
- **Advanced OCR**: High-accuracy text extraction using PaddleOCR with layout preservation
- **Receipt Parsing**: Intelligent extraction of structured data from receipts
- **Table Detection**: Automatic detection and structuring of tabular data
- **Batch Processing**: Process multiple receipt images simultaneously
- **Spending Analysis**: Comprehensive financial analysis and reporting

### Key Capabilities
- ✨ **Smart Text Extraction** with preserved spacing and layout
- 🧾 **Receipt Information Extraction** (merchant, items, prices, dates, etc.)
- 📊 **Table Structure Recognition** and formatting
- 📦 **Batch Processing** for multiple images
- 📈 **Spending Analytics** with charts and insights
- 💾 **Multiple Export Formats** (JSON, CSV, Excel)
- 🔍 **Receipt Search** and duplicate detection
- 📱 **Web Interface** built with Streamlit

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Features Overview](#features-overview)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/receipt-ocr-system.git](https://github.com/AHMEDSANA/Receipt-OCR-and-Analysis-System.git)
   cd receipt-ocr-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Required Packages

```text
streamlit>=1.28.0
paddleocr>=2.7.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
spacy>=3.6.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
openpyxl>=3.1.0
sqlite3
```

## 🚀 Quick Start

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run PaddleOCR_streamlit.py
```

Open your browser to `http://localhost:8501` to access the interface.

### Command Line Usage

```python
from receipt_extractor import ReceiptExtractor
from receipt_analyzer import ReceiptAnalyzer

# Initialize extractor
extractor = ReceiptExtractor()

# Process a single receipt
receipt_info = extractor.extract_receipt_info(ocr_text)
print(extractor.format_receipt_summary(receipt_info))

# Batch processing
from batch_processor import BatchReceiptProcessor
processor = BatchReceiptProcessor()
results = processor.process_images_folder("/path/to/images")
```

## 📖 Usage

### Single Image Processing

1. Upload an image through the web interface
2. Configure processing options:
   - Enable debug mode for detailed processing steps
   - Toggle table detection
   - Choose spacing preservation settings
3. Click "Process Image" to extract text and data
4. View results in organized tabs
5. Download extracted data in your preferred format

### Batch Processing

Process multiple receipts at once:

1. Select "Batch Processing" mode
2. Upload multiple files or specify a folder path
3. Configure extraction settings
4. Review processing results and statistics
5. Export consolidated data

### Analysis Dashboard

Track your spending patterns:

1. Navigate to "Analysis Dashboard"
2. View spending summaries and trends
3. Filter by date ranges
4. Explore merchant and category breakdowns
5. Generate detailed reports

## 🎯 Features Overview

### Receipt Information Extraction

The system can extract comprehensive receipt data including:

- **Merchant Information**: Name, address, phone, email
- **Transaction Details**: Date, time, receipt number, cashier ID
- **Financial Data**: Subtotal, tax, discounts, tips, total
- **Payment Info**: Method, card type, last four digits
- **Line Items**: Individual products with prices and quantities

### Advanced OCR Features

- **Layout Preservation**: Maintains original text spacing and structure
- **Table Detection**: Identifies and structures tabular data
- **Image Preprocessing**: Automatic deskewing, noise removal, contrast enhancement
- **Multi-format Support**: PNG, JPG, JPEG, BMP, TIFF

### Data Management

- **SQLite Database**: Local storage for processed receipts
- **Search Functionality**: Find receipts by merchant, amount, or content
- **Duplicate Detection**: Identify potential duplicate entries
- **Export Options**: JSON, CSV, Excel formats

## ⚙️ Configuration

### OCR Settings

Customize OCR behavior in [`receipt_config.py`](receipt_config.py):

```python
OCR_SETTINGS = {
    'use_angle_cls': True,
    'lang': 'en',
    'use_gpu': False,
    'det_db_thresh': 0.3,
    # ... additional settings
}
```

### Extraction Patterns

Configure extraction patterns for different receipt elements:

```python
EXTRACTION_PATTERNS = {
    'date': [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        # ... more patterns
    ],
    'price': [
        r'\$(\d+\.\d{2})',
        r'(\d+\.\d{2})\s*\$',
        # ... more patterns
    ]
}
```

### Merchant-Specific Rules

Add custom rules for specific merchants:

```python
MERCHANT_SPECIFIC_RULES = {
    'walmart': {
        'total_indicators': ['total', 'balance due'],
        'tax_indicators': ['tax'],
        # ... additional rules
    }
}
```

## 📚 API Reference

### Core Classes

#### `ReceiptExtractor`
Main class for receipt information extraction.

```python
extractor = ReceiptExtractor(nlp_model=None)
receipt_info = extractor.extract_receipt_info(text)
summary = extractor.format_receipt_summary(receipt_info)
```

#### `BatchReceiptProcessor`
Handle multiple receipt processing.

```python
processor = BatchReceiptProcessor(debug_mode=False)
results = processor.process_images_folder(folder_path)
results = processor.process_uploaded_files(uploaded_files)
```

#### `ReceiptAnalyzer`
Analyze and manage receipt data.

```python
analyzer = ReceiptAnalyzer(db_path="receipts.db")
analyzer.save_receipt(receipt_info)
report = analyzer.generate_spending_report()
```

### Data Structures

#### `ReceiptInfo`
Comprehensive receipt information container:

```python
@dataclass
class ReceiptInfo:
    merchant_name: Optional[str] = None
    transaction_date: Optional[str] = None
    total: Optional[float] = None
    items: List[ReceiptItem] = None
    # ... additional fields
```

#### `ReceiptItem`
Individual receipt item information:

```python
@dataclass
class ReceiptItem:
    name: str
    price: float
    quantity: Optional[int] = None
    category: Optional[str] = None
```

## 💡 Examples

### Extract Receipt Information

```python
from receipt_extractor import ReceiptExtractor
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize extractor
extractor = ReceiptExtractor(nlp)

# Sample OCR text
ocr_text = """
WALMART SUPERCENTER
123 MAIN ST
ANYTOWN, ST 12345

Date: 2024-01-15
Time: 14:30

Bananas           $2.99
Milk 2%           $3.49
Bread             $2.29

Subtotal          $8.77
Tax               $0.70
TOTAL             $9.47

VISA ****1234
"""

# Extract information
receipt = extractor.extract_receipt_info(ocr_text)

# Display results
print(f"Merchant: {receipt.merchant_name}")
print(f"Total: ${receipt.total:.2f}")
print(f"Items: {len(receipt.items)}")
for item in receipt.items:
    print(f"  - {item.name}: ${item.price:.2f}")
```

### Batch Processing

```python
from batch_processor import BatchReceiptProcessor

# Initialize processor
processor = BatchReceiptProcessor()

# Process folder of images
results = processor.process_images_folder("/path/to/receipt/images")

# Display summary
summary = results['summary']
print(f"Processed: {summary['total_files_processed']} files")
print(f"Success rate: {summary['extraction_success_rate']:.1%}")
print(f"Total amount: ${summary['total_amount_found']:.2f}")

# Export results
processor.export_batch_results(results, 'excel')
```

### Spending Analysis

```python
from receipt_analyzer import ReceiptAnalyzer

# Initialize analyzer
analyzer = ReceiptAnalyzer()

# Generate spending report
report = analyzer.generate_spending_report(
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# Display insights
print(f"Total spent: ${report['summary']['total_spent']:.2f}")
print(f"Average transaction: ${report['summary']['average_transaction']:.2f}")

# Top merchants
for merchant in report['merchant_spending'][:5]:
    print(f"{merchant['merchant']}: ${merchant['total_spent']:.2f}")
```

## 🎨 Web Interface Screenshots

The Streamlit interface provides:

- **Single Image Processing**: Upload and process individual receipts
- **Batch Processing**: Handle multiple images simultaneously
- **Analysis Dashboard**: Visualize spending patterns and trends
- **Interactive Charts**: Explore data with dynamic visualizations
- **Export Tools**: Download data in multiple formats

## 🔧 Advanced Configuration

### Custom Extraction Patterns

Add new extraction patterns in [`receipt_config.py`](receipt_config.py):

```python
# Add custom patterns
ReceiptConfig.update_pattern('custom_field', [
    r'pattern1',
    r'pattern2'
])

# Add custom keywords
ReceiptConfig.update_keywords('custom_category', [
    'keyword1', 'keyword2'
])
```

### Database Configuration

Customize database settings:

```python
DATABASE_SETTINGS = {
    'default_db_path': 'receipts.db',
    'enable_full_text_search': True,
    'auto_backup': True,
    'backup_frequency_days': 7
}
```

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **PaddleOCR installation issues**
   ```bash
   pip install paddlepaddle paddleocr
   ```

3. **Image processing errors**
   - Ensure images are in supported formats (PNG, JPG, JPEG, BMP, TIFF)
   - Check image file permissions
   - Verify image files are not corrupted

4. **Low extraction accuracy**
   - Use high-resolution images
   - Ensure good contrast
   - Enable image preprocessing options

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/receipt-ocr-system.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for OCR capabilities
- [spaCy](https://spacy.io/) for natural language processing
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for image processing

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/receipt-ocr-system/issues) page
2. Create a new issue with detailed information
3. Include sample images and error messages when possible

## 🗺️ Roadmap

- [ ] Mobile app development
- [ ] Cloud deployment options
- [ ] Advanced ML models for better accuracy
- [ ] Multi-language support
- [ ] API endpoint development
- [ ] Integration with accounting software

---

**⭐ If you find this project helpful, please give it a star!**
