# filepath: d:\Hexlar\Reciept File\Paddle_OCR\receipt_extractor.py
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date
import spacy
from dataclasses import dataclass, asdict
import streamlit as st


@dataclass
class ReceiptItem:
    """Represents a single item on a receipt."""
    name: str
    price: float
    quantity: Optional[int] = None
    unit_price: Optional[float] = None
    category: Optional[str] = None
    line_number: Optional[int] = None
    

@dataclass
class ReceiptInfo:
    """Comprehensive receipt information structure."""
    # Merchant Information
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    merchant_phone: Optional[str] = None
    merchant_email: Optional[str] = None
    
    # Receipt Details
    receipt_number: Optional[str] = None
    transaction_id: Optional[str] = None
    cashier_id: Optional[str] = None
    terminal_id: Optional[str] = None
    
    # Date and Time
    transaction_date: Optional[str] = None
    transaction_time: Optional[str] = None
    
    # Items
    items: List[ReceiptItem] = None
    
    # Financial Information
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    tax_rate: Optional[float] = None
    discount: Optional[float] = None
    tip: Optional[float] = None
    total: Optional[float] = None
    
    # Payment Information
    payment_method: Optional[str] = None
    card_type: Optional[str] = None
    card_last_four: Optional[str] = None
    change_given: Optional[float] = None
    amount_paid: Optional[float] = None
    
    # Additional Info
    currency: Optional[str] = "USD"
    receipt_type: Optional[str] = None  # retail, restaurant, gas, etc.
    raw_text: Optional[str] = None
    confidence_score: Optional[float] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []


class ReceiptExtractor:
    """Advanced receipt information extractor using NLP and pattern matching."""
    
    def __init__(self, nlp_model=None):
        """Initialize the receipt extractor."""
        self.nlp = nlp_model or self._load_nlp_model()
        self._setup_patterns()
        self._setup_keywords()
        
    @st.cache_resource
    def _load_nlp_model(_self):
        """Load spaCy NLP model."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
    
    def _setup_patterns(self):
        """Setup regex patterns for different receipt elements."""
        self.patterns = {
            # Date patterns
            'date': [
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'(\w+ \d{1,2},? \d{4})',
                r'(\d{1,2} \w+ \d{4})',
            ],
            
            # Time patterns
            'time': [
                r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',
                r'(\d{1,2}\.\d{2}(?:\.\d{2})?)',
            ],
            
            # Price patterns
            'price': [
                r'(\$?\d+\.\d{2})',
                r'(\d+,\d{3}\.\d{2})',
                r'(\d+\.\d{2}\s*\$)',
            ],
            
            # Receipt/Transaction numbers
            'receipt_number': [
                r'(?:receipt|rcpt|trans|transaction)[\s#:]*(\w+\d+|\d+)',
                r'(?:ref|reference)[\s#:]*(\w+)',
                r'(?:order|ord)[\s#:]*(\w+\d+)',
            ],
            
            # Phone numbers
            'phone': [
                r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
                r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            ],
            
            # Email addresses
            'email': [
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            ],
            
            # Card information
            'card_last_four': [
                r'(?:xxxx|****)\s*(\d{4})',
                r'(?:ending|last)\s*(?:in\s*)?(\d{4})',
            ],
            
            # Tax patterns
            'tax': [
                r'(?:tax|hst|gst|vat)[\s:]*\$?(\d+\.\d{2})',
                r'(\d+\.\d{2})%?\s*(?:tax|hst|gst|vat)',
            ],
        }
    
    def _setup_keywords(self):
        """Setup keywords for different categories."""
        self.keywords = {
            'totals': ['total', 'amount', 'grand total', 'balance', 'sum'],
            'subtotals': ['subtotal', 'sub total', 'sub-total', 'before tax'],
            'discounts': ['discount', 'savings', 'coupon', 'promo', 'off'],
            'tax': ['tax', 'hst', 'gst', 'vat', 'sales tax'],
            'payment_methods': ['cash', 'credit', 'debit', 'visa', 'mastercard', 'amex', 'discover', 'paypal'],
            'receipt_types': ['retail', 'restaurant', 'gas', 'grocery', 'pharmacy', 'hotel'],
        }
    
    def extract_receipt_info(self, text: str) -> ReceiptInfo:
        """
        Extract comprehensive receipt information from OCR text.
        
        Args:
            text: Raw OCR text from receipt
            
        Returns:
            ReceiptInfo object with extracted information
        """
        receipt = ReceiptInfo(raw_text=text)
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        lines = cleaned_text.split('\n')
        
        # Extract basic information
        receipt.merchant_name = self._extract_merchant_name(lines)
        receipt.merchant_address = self._extract_address(cleaned_text)
        receipt.merchant_phone = self._extract_phone(cleaned_text)
        receipt.merchant_email = self._extract_email(cleaned_text)
        
        # Extract date and time
        receipt.transaction_date = self._extract_date(cleaned_text)
        receipt.transaction_time = self._extract_time(cleaned_text)
        
        # Extract receipt identifiers
        receipt.receipt_number = self._extract_receipt_number(cleaned_text)
        
        # Extract financial information
        receipt.total = self._extract_total(lines)
        receipt.subtotal = self._extract_subtotal(lines)
        receipt.tax_amount = self._extract_tax(lines)
        receipt.discount = self._extract_discount(lines)
        
        # Extract payment information
        receipt.payment_method = self._extract_payment_method(cleaned_text)
        receipt.card_last_four = self._extract_card_last_four(cleaned_text)
        
        # Extract items
        receipt.items = self._extract_items(lines)
        
        # Determine receipt type
        receipt.receipt_type = self._classify_receipt_type(cleaned_text)
        
        # Calculate confidence score
        receipt.confidence_score = self._calculate_confidence(receipt)
        
        return receipt
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize the text."""
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            if line:  # Skip empty lines
                # Normalize multiple spaces to single space
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_merchant_name(self, lines: List[str]) -> Optional[str]:
        """Extract merchant name from the first few lines."""
        # Usually merchant name is in the first 1-3 lines
        for i, line in enumerate(lines[:5]):
            line = line.strip().upper()
            
            # Skip lines that look like addresses or numbers
            if (re.search(r'\d{3,}', line) or 
                re.search(r'(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD)', line) or
                len(line) < 3):
                continue
            
            # Skip common non-merchant terms
            skip_terms = ['RECEIPT', 'INVOICE', 'CUSTOMER', 'COPY', 'THANK YOU']
            if any(term in line for term in skip_terms):
                continue
                
            return line.title()
        
        return None
    
    def _extract_address(self, text: str) -> Optional[str]:
        """Extract merchant address."""
        # Look for address patterns
        address_patterns = [
            r'(\d+\s+\w+(?:\s+\w+)*\s+(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|RD|ROAD|DR|DRIVE|LN|LANE))',
            r'(\d+\s+[A-Za-z\s]+(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|RD|ROAD))',
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number."""
        for pattern in self.patterns['phone']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address."""
        for pattern in self.patterns['email']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract transaction date."""
        for pattern in self.patterns['date']:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                # Try to normalize the date format
                try:
                    # Parse different date formats and standardize
                    parsed_date = self._parse_date(date_str)
                    if parsed_date:
                        return parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
                return date_str
        return None
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string into date object."""
        date_formats = [
            '%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None
    
    def _extract_time(self, text: str) -> Optional[str]:
        """Extract transaction time."""
        for pattern in self.patterns['time']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _extract_receipt_number(self, text: str) -> Optional[str]:
        """Extract receipt or transaction number."""
        for pattern in self.patterns['receipt_number']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_total(self, lines: List[str]) -> Optional[float]:
        """Extract total amount."""
        # Look for total in the last few lines
        for line in reversed(lines[-10:]):
            line_lower = line.lower()
            
            # Check if line contains total keywords
            if any(keyword in line_lower for keyword in self.keywords['totals']):
                # Extract price from this line
                price = self._extract_price_from_line(line)
                if price is not None:
                    return price
        
        return None
    
    def _extract_subtotal(self, lines: List[str]) -> Optional[float]:
        """Extract subtotal amount."""
        for line in lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in self.keywords['subtotals']):
                price = self._extract_price_from_line(line)
                if price is not None:
                    return price
        
        return None
    
    def _extract_tax(self, lines: List[str]) -> Optional[float]:
        """Extract tax amount."""
        for line in lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in self.keywords['tax']):
                price = self._extract_price_from_line(line)
                if price is not None:
                    return price
        
        return None
    
    def _extract_discount(self, lines: List[str]) -> Optional[float]:
        """Extract discount amount."""
        for line in lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in self.keywords['discounts']):
                price = self._extract_price_from_line(line)
                if price is not None:
                    return price
        
        return None
    
    def _extract_price_from_line(self, line: str) -> Optional[float]:
        """Extract price value from a line."""
        # Remove currency symbols and clean the line
        cleaned_line = re.sub(r'[^\d\.\-,]', ' ', line)
        
        # Find all potential price matches
        price_matches = re.findall(r'\d+\.\d{2}', cleaned_line)
        
        if price_matches:
            # Return the last (usually rightmost) price found
            try:
                return float(price_matches[-1])
            except ValueError:
                pass
        
        return None
    
    def _extract_payment_method(self, text: str) -> Optional[str]:
        """Extract payment method."""
        text_lower = text.lower()
        
        for method in self.keywords['payment_methods']:
            if method in text_lower:
                return method.upper()
        
        return None
    
    def _extract_card_last_four(self, text: str) -> Optional[str]:
        """Extract the last four digits of a payment card."""
        patterns = [
            r"card\s*[#x*].*?(\d{4})",
            r"credit\s*card.*?(\d{4})",
            r"credit.*?x+(\d{4})",
            r"card.*?x+(\d{4})",
            r"(?:visa|mastercard|amex|discover).*?(\d{4})",
            r"(?:\*{4}|\d{4})[- ]*(?:\*{4}|\d{4})[- ]*(?:\*{4}|\d{4})[- ]*(\d{4})",
            r"x+(\d{4})",
            r"\*+(\d{4})",
            r"[xX*]{1,10}(\d{4})",  # Fixed pattern that was causing an error
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_items(self, lines: List[str]) -> List[ReceiptItem]:
        """Extract individual items from receipt."""
        items = []
        
        # Define patterns that typically indicate start/end of items section
        start_indicators = ['item', 'product', 'description', 'qty']
        end_indicators = ['subtotal', 'sub total', 'total', 'tax', 'amount due']
        
        # Find the section containing items
        item_section_start = 0
        item_section_end = len(lines)
        
        # Look for subtotal or total to determine where items end
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in end_indicators):
                if self._extract_price_from_line(line):  # Confirm it has a price
                    item_section_end = i
                    break
        
        # Extract items from the identified section
        for i in range(item_section_start, min(item_section_end, len(lines))):
            line = lines[i].strip()
            if not line or len(line) < 3:
                continue
            
            # Skip lines that are clearly not items
            line_lower = line.lower()
            if (any(keyword in line_lower for keyword in ['thank you', 'welcome', 'customer', 'cashier']) or
                re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', line) or  # Date line
                re.match(r'^\d{1,2}:\d{2}', line)):  # Time line
                continue
            
            # Try to extract item information
            item = self._parse_item_line(line, i)
            if item:
                items.append(item)
        
        return items
    
    def _parse_item_line(self, line: str, line_number: int) -> Optional[ReceiptItem]:
        """Parse a single line to extract item information."""
        # Extract price from the line
        price = self._extract_price_from_line(line)
        if price is None:
            return None
        
        # Remove price from line to get item name
        # Find the price in the original line and remove it
        price_pattern = r'\$?\d+\.\d{2}'
        matches = list(re.finditer(price_pattern, line))
        
        if matches:
            # Remove the last price match (assume it's the item price)
            last_match = matches[-1]
            item_name = (line[:last_match.start()] + line[last_match.end():]).strip()
        else:
            item_name = line.strip()
        
        # Clean item name
        item_name = re.sub(r'\s+', ' ', item_name).strip()
        
        # Skip if name is too short or looks like a total line
        if len(item_name) < 2:
            return None
        
        # Try to extract quantity if present
        quantity = self._extract_quantity_from_name(item_name)
        if quantity:
            # Remove quantity from name
            item_name = re.sub(r'\b\d+\s*x\s*|\b\d+\s*@\s*', '', item_name, flags=re.IGNORECASE).strip()
        
        return ReceiptItem(
            name=item_name,
            price=price,
            quantity=quantity,
            line_number=line_number
        )
    
    def _extract_quantity_from_name(self, name: str) -> Optional[int]:
        """Extract quantity from item name."""
        # Look for patterns like "2x", "3 @", etc.
        qty_patterns = [
            r'(\d+)\s*x\s*',
            r'(\d+)\s*@\s*',
            r'qty:?\s*(\d+)',
        ]
        
        for pattern in qty_patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        return None
    
    def _classify_receipt_type(self, text: str) -> Optional[str]:
        """Classify the type of receipt."""
        text_lower = text.lower()
        
        # Define keywords for different receipt types
        type_keywords = {
            'restaurant': ['restaurant', 'cafe', 'dine', 'server', 'tip', 'gratuity', 'table'],
            'grocery': ['grocery', 'market', 'produce', 'deli', 'bakery', 'organic'],
            'gas': ['gas', 'fuel', 'gallons', 'pump', 'station'],
            'pharmacy': ['pharmacy', 'rx', 'prescription', 'drug', 'cvs', 'walgreens'],
            'retail': ['store', 'shop', 'retail', 'clothing', 'department'],
        }
        
        # Count matches for each type
        type_scores = {}
        for receipt_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                type_scores[receipt_type] = score
        
        # Return the type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return 'retail'  # Default
    
    def _calculate_confidence(self, receipt: ReceiptInfo) -> float:
        """Calculate confidence score based on extracted information."""
        score = 0.0
        total_checks = 10
        
        # Check for key fields
        if receipt.merchant_name:
            score += 1
        if receipt.total is not None:
            score += 2  # Total is very important
        if receipt.transaction_date:
            score += 1
        if receipt.items:
            score += 2  # Items are important
        if receipt.subtotal is not None:
            score += 0.5
        if receipt.tax_amount is not None:
            score += 0.5
        if receipt.payment_method:
            score += 0.5
        if receipt.receipt_number:
            score += 0.5
        if receipt.merchant_address:
            score += 0.5
        if receipt.transaction_time:
            score += 0.5
        
        return min(score / total_checks, 1.0)
    
    def format_receipt_summary(self, receipt: ReceiptInfo) -> str:
        """Format receipt information into a readable summary."""
        summary = []
        
        # Header
        summary.append("=== RECEIPT SUMMARY ===\n")
        
        # Merchant Information
        if receipt.merchant_name:
            summary.append(f"Merchant: {receipt.merchant_name}")
        if receipt.merchant_address:
            summary.append(f"Address: {receipt.merchant_address}")
        if receipt.merchant_phone:
            summary.append(f"Phone: {receipt.merchant_phone}")
        
        summary.append("")  # Empty line
        
        # Transaction Details
        if receipt.transaction_date:
            summary.append(f"Date: {receipt.transaction_date}")
        if receipt.transaction_time:
            summary.append(f"Time: {receipt.transaction_time}")
        if receipt.receipt_number:
            summary.append(f"Receipt #: {receipt.receipt_number}")
        
        summary.append("")  # Empty line
        
        # Items
        if receipt.items:
            summary.append("ITEMS:")
            for i, item in enumerate(receipt.items, 1):
                qty_str = f"{item.quantity}x " if item.quantity else ""
                summary.append(f"  {i}. {qty_str}{item.name} - ${item.price:.2f}")
        
        summary.append("")  # Empty line
        
        # Totals
        if receipt.subtotal is not None:
            summary.append(f"Subtotal: ${receipt.subtotal:.2f}")
        if receipt.tax_amount is not None:
            summary.append(f"Tax: ${receipt.tax_amount:.2f}")
        if receipt.discount is not None:
            summary.append(f"Discount: -${receipt.discount:.2f}")
        if receipt.total is not None:
            summary.append(f"TOTAL: ${receipt.total:.2f}")
        
        summary.append("")  # Empty line
        
        # Payment Information
        if receipt.payment_method:
            summary.append(f"Payment: {receipt.payment_method}")
        if receipt.card_last_four:
            summary.append(f"Card ending in: {receipt.card_last_four}")
        
        # Metadata
        if receipt.receipt_type:
            summary.append(f"\nReceipt Type: {receipt.receipt_type}")
        if receipt.confidence_score is not None:
            summary.append(f"Confidence: {receipt.confidence_score:.1%}")
        
        return "\n".join(summary)
    
    def export_to_json(self, receipt: ReceiptInfo) -> str:
        """Export receipt information to JSON format."""
        # Convert dataclass to dictionary
        receipt_dict = asdict(receipt)
        
        # Add metadata
        receipt_dict['exported_at'] = datetime.now().isoformat()
        receipt_dict['extractor_version'] = "1.0"
        
        return json.dumps(receipt_dict, indent=2, ensure_ascii=False)
    
    def export_to_csv_row(self, receipt: ReceiptInfo) -> Dict[str, Any]:
        """Export receipt information as a dictionary suitable for CSV export."""
        # Calculate totals
        items_total = sum(item.price for item in receipt.items) if receipt.items else 0
        items_count = len(receipt.items) if receipt.items else 0
        
        return {
            'merchant_name': receipt.merchant_name or '',
            'transaction_date': receipt.transaction_date or '',
            'transaction_time': receipt.transaction_time or '',
            'receipt_number': receipt.receipt_number or '',
            'total': receipt.total or 0,
            'subtotal': receipt.subtotal or 0,
            'tax_amount': receipt.tax_amount or 0,
            'discount': receipt.discount or 0,
            'items_count': items_count,
            'items_total': items_total,
            'payment_method': receipt.payment_method or '',
            'receipt_type': receipt.receipt_type or '',
            'confidence_score': receipt.confidence_score or 0,
            'processed_at': datetime.now().isoformat()
        }