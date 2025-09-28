# filepath: d:\Hexlar\Reciept File\Paddle_OCR\receipt_config.py
"""
Configuration file for receipt extraction system.
Contains patterns, keywords, and settings for customizing extraction behavior.
"""

import re
from typing import Dict, List, Any, Optional


class ReceiptConfig:
    """Configuration class for receipt extraction settings."""
    
    # OCR Settings
    OCR_SETTINGS = {
        'use_angle_cls': True,
        'lang': 'en',
        'use_gpu': False,
        'det_db_thresh': 0.3,
        'det_db_box_thresh': 0.6,
        'det_db_unclip_ratio': 1.5,
        'det_east_score_thresh': 0.8,
        'det_east_cover_thresh': 0.1,
        'det_east_nms_thresh': 0.2,
        'rec_batch_num': 6,
        'max_text_length': 25,
        'rec_char_dict_path': None,
        'use_space_char': True,
        'vis_font_path': None,
        'drop_score': 0.5,
        'enable_mkldnn': False,
        'cpu_threads': 10,
        'use_pdserving': False,
        'warmup': False,
        'draw_img_save_dir': './inference_results',
        'save_crop_res': False,
        'crop_res_save_dir': './output',
        'use_mp': False,
        'total_process_num': 1,
        'process_id': 0,
        'benchmark': False,
        'save_log_path': './log_output/',
        'show_log': True,
        'use_onnx': False
    }
    
    # Text preprocessing settings
    TEXT_PREPROCESSING = {
        'remove_excessive_spaces': True,
        'normalize_line_breaks': True,
        'preserve_table_structure': True,
        'min_line_length': 1,
        'max_consecutive_spaces': 5
    }
    
    # Image preprocessing settings
    IMAGE_PREPROCESSING = {
        'apply_deskewing': True,
        'skew_angle_threshold': 0.5,
        'apply_noise_removal': True,
        'bilateral_filter_params': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
        'apply_contrast_enhancement': True,
        'clahe_params': {'clipLimit': 2.0, 'tileGridSize': (8, 8)},
        'apply_adaptive_threshold': True,
        'threshold_params': {
            'maxValue': 255,
            'adaptiveMethod': 'ADAPTIVE_THRESH_GAUSSIAN_C',
            'thresholdType': 'THRESH_BINARY',
            'blockSize': 11,
            'C': 2
        },
        'apply_morphological_operations': True,
        'morphology_kernel_size': (1, 1)
    }
    
    # Layout analysis settings
    LAYOUT_ANALYSIS = {
        'line_threshold_ratio': 0.02,  # 2% of image height
        'character_width_estimation': True,
        'preserve_spacing': True,
        'max_spaces_between_words': 25,
        'min_gap_for_spacing': 0.5  # Multiplier of character width
    }
    
    # Table detection settings
    TABLE_DETECTION = {
        'enable_table_detection': True,
        'min_table_area': 5000,
        'aspect_ratio_range': (0.2, 5.0),
        'row_threshold_ratio': 0.03,  # 3% of image height
        'column_threshold_ratio': 0.05,  # 5% of image width
        'cell_padding': 2
    }
    
    # Receipt extraction patterns
    EXTRACTION_PATTERNS = {
        'date': [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(\w+ \d{1,2},? \d{4})',
            r'(\d{1,2} \w+ \d{4})',
            r'(\d{1,2}[/.]\d{1,2}[/.]\d{2,4})',
        ],
        
        'time': [
            r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',
            r'(\d{1,2}\.\d{2}(?:\.\d{2})?)',
            r'(\d{1,2}h\d{2}m?)',
        ],
        
        'price': [
            r'\$(\d+\.\d{2})',
            r'(\d+\.\d{2})\s*\$',
            r'(\d+,\d{3}\.\d{2})',
            r'USD\s*(\d+\.\d{2})',
            r'(\d+\.\d{2})\s*USD',
        ],
        
        'receipt_number': [
            r'(?:receipt|rcpt|trans|transaction)[\s#:]*(\w*\d+\w*)',
            r'(?:ref|reference)[\s#:]*(\w+)',
            r'(?:order|ord)[\s#:]*(\w*\d+\w*)',
            r'(?:invoice|inv)[\s#:]*(\w*\d+\w*)',
            r'(?:ticket|tkt)[\s#:]*(\w*\d+\w*)',
        ],
        
        'phone': [
            r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            r'(\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        ],
        
        'email': [
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        ],
        
        'card_last_four': [
            r'(?:xxxx|****|\*{4})\s*(\d{4})',
            r'(?:ending|last)\s*(?:in\s*)?(\d{4})',
            r'(?:card)\s*(?:ending\s*)?(?:in\s*)?(\d{4})',
        ],
        
        'tax': [
            r'(?:tax|hst|gst|vat|sales\s*tax)[\s:]*\$?(\d+\.\d{2})',
            r'(\d+\.\d{2})%?\s*(?:tax|hst|gst|vat)',
            r'(?:tax|hst|gst|vat)\s*@\s*[\d.]+%[\s:]*\$?(\d+\.\d{2})',
        ],
        
        'discount': [
            r'(?:discount|savings|off|coupon|promo)[\s:]*\$?(\d+\.\d{2})',
            r'(\d+\.\d{2})\s*(?:discount|off|savings)',
            r'(?:save|saved)[\s:]*\$?(\d+\.\d{2})',
        ],
        
        'tip': [
            r'(?:tip|gratuity)[\s:]*\$?(\d+\.\d{2})',
            r'(\d+\.\d{2})\s*(?:tip|gratuity)',
        ],
    }
    
    # Keywords for different categories
    KEYWORDS = {
        'totals': [
            'total', 'amount', 'grand total', 'balance', 'sum',
            'amount due', 'total due', 'final total', 'net total'
        ],
        
        'subtotals': [
            'subtotal', 'sub total', 'sub-total', 'before tax',
            'pre-tax', 'net amount', 'merchandise total'
        ],
        
        'discounts': [
            'discount', 'savings', 'coupon', 'promo', 'off',
            'markdown', 'sale', 'reduction', 'deduction'
        ],
        
        'tax': [
            'tax', 'hst', 'gst', 'vat', 'sales tax', 'state tax',
            'city tax', 'local tax', 'excise tax'
        ],
        
        'payment_methods': [
            'cash', 'credit', 'debit', 'visa', 'mastercard', 'amex',
            'american express', 'discover', 'paypal', 'venmo',
            'apple pay', 'google pay', 'contactless'
        ],
        
        'receipt_types': [
            'retail', 'restaurant', 'gas', 'grocery', 'pharmacy',
            'hotel', 'service', 'entertainment', 'transport'
        ],
        
        'merchant_indicators': [
            'store', 'shop', 'restaurant', 'cafe', 'market',
            'pharmacy', 'station', 'hotel', 'company', 'inc',
            'llc', 'ltd', 'corp'
        ],
        
        'skip_lines': [
            'thank you', 'welcome', 'customer copy', 'merchant copy',
            'signature', 'approved', 'please sign', 'retain receipt',
            'customer service', 'return policy'
        ]
    }
    
    # Merchant name extraction settings
    MERCHANT_EXTRACTION = {
        'max_lines_to_check': 5,
        'min_name_length': 3,
        'skip_address_lines': True,
        'skip_numeric_lines': True,
        'common_suffixes': ['inc', 'llc', 'ltd', 'corp', 'co'],
        'address_indicators': [
            'st', 'street', 'ave', 'avenue', 'blvd', 'boulevard',
            'rd', 'road', 'dr', 'drive', 'ln', 'lane', 'way',
            'plaza', 'square', 'circle'
        ]
    }
    
    # Item extraction settings
    ITEM_EXTRACTION = {
        'enable_quantity_detection': True,
        'quantity_patterns': [
            r'(\d+)\s*x\s*',
            r'(\d+)\s*@\s*',
            r'qty:?\s*(\d+)',
            r'quantity:?\s*(\d+)',
        ],
        'price_position': 'right',  # 'left', 'right', 'auto'
        'min_item_name_length': 2,
        'max_item_name_length': 100,
        'skip_total_lines': True,
        'category_keywords': {
            'food': ['pizza', 'burger', 'sandwich', 'salad', 'soup', 'drink'],
            'grocery': ['milk', 'bread', 'eggs', 'fruit', 'vegetable'],
            'fuel': ['gas', 'gasoline', 'diesel', 'unleaded'],
            'pharmacy': ['rx', 'prescription', 'medicine', 'vitamin'],
            'clothing': ['shirt', 'pants', 'dress', 'shoes', 'jacket']
        }
    }
    
    # Date parsing formats
    DATE_FORMATS = [
        '%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y',
        '%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y',
        '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
        '%m.%d.%Y', '%d.%m.%Y', '%Y.%m.%d'
    ]
    
    # Confidence scoring weights
    CONFIDENCE_WEIGHTS = {
        'merchant_name': 1.0,
        'total': 2.0,
        'transaction_date': 1.0,
        'items': 2.0,
        'subtotal': 0.5,
        'tax_amount': 0.5,
        'payment_method': 0.5,
        'receipt_number': 0.5,
        'merchant_address': 0.5,
        'transaction_time': 0.5
    }
    
    # Export settings
    EXPORT_SETTINGS = {
        'csv_delimiter': ',',
        'csv_encoding': 'utf-8',
        'excel_engine': 'openpyxl',
        'json_indent': 2,
        'include_raw_text': True,
        'include_confidence_scores': True,
        'date_format': '%Y-%m-%d',
        'time_format': '%H:%M:%S'
    }
    
    # Database settings
    DATABASE_SETTINGS = {
        'default_db_path': 'receipts.db',
        'enable_full_text_search': True,
        'auto_backup': True,
        'backup_frequency_days': 7,
        'max_backups_to_keep': 10
    }
    
    # Analysis settings
    ANALYSIS_SETTINGS = {
        'default_chart_style': 'seaborn',
        'color_palette': 'viridis',
        'figure_size': (10, 6),
        'dpi': 100,
        'enable_interactive_plots': True,
        'max_categories_in_chart': 15
    }
    
    @classmethod
    def get_pattern(cls, pattern_type: str) -> List[str]:
        """Get extraction patterns for a specific type."""
        return cls.EXTRACTION_PATTERNS.get(pattern_type, [])
    
    @classmethod
    def get_keywords(cls, keyword_type: str) -> List[str]:
        """Get keywords for a specific category."""
        return cls.KEYWORDS.get(keyword_type, [])
    
    @classmethod
    def update_pattern(cls, pattern_type: str, new_patterns: List[str]):
        """Update extraction patterns for a specific type."""
        if pattern_type in cls.EXTRACTION_PATTERNS:
            cls.EXTRACTION_PATTERNS[pattern_type].extend(new_patterns)
    
    @classmethod
    def update_keywords(cls, keyword_type: str, new_keywords: List[str]):
        """Update keywords for a specific category."""
        if keyword_type in cls.KEYWORDS:
            cls.KEYWORDS[keyword_type].extend(new_keywords)
    
    @classmethod
    def get_ocr_settings(cls) -> Dict[str, Any]:
        """Get OCR configuration settings."""
        return cls.OCR_SETTINGS.copy()
    
    @classmethod
    def get_preprocessing_settings(cls) -> Dict[str, Any]:
        """Get image preprocessing settings."""
        return cls.IMAGE_PREPROCESSING.copy()
    
    @classmethod
    def get_confidence_threshold(cls, field_type: str = 'default') -> float:
        """Get confidence threshold for a specific field type."""
        thresholds = {
            'merchant_name': 0.6,
            'total': 0.8,
            'items': 0.7,
            'date': 0.7,
            'default': 0.6
        }
        return thresholds.get(field_type, thresholds['default'])


# Custom extraction rules for specific merchant types
MERCHANT_SPECIFIC_RULES = {
    'walmart': {
        'total_indicators': ['total', 'balance due'],
        'tax_indicators': ['tax'],
        'item_section_start': ['qty', 'item'],
        'item_section_end': ['subtotal']
    },
    
    'target': {
        'total_indicators': ['total', 'amount due'],
        'tax_indicators': ['sales tax', 'tax'],
        'discount_indicators': ['target circle', 'cartwheel']
    },
    
    'starbucks': {
        'receipt_type': 'restaurant',
        'total_indicators': ['total', 'amount'],
        'tax_indicators': ['tax'],
        'tip_indicators': ['tip', 'gratuity']
    },
    
    'mcdonalds': {
        'receipt_type': 'restaurant',
        'total_indicators': ['total', 'amount due'],
        'tax_indicators': ['tax']
    },
    
    'shell': {
        'receipt_type': 'gas',
        'total_indicators': ['total', 'amount'],
        'fuel_indicators': ['gallons', 'gal', 'unleaded', 'premium']
    }
}


def get_merchant_rules(merchant_name: str) -> Optional[Dict[str, Any]]:
    """Get specific extraction rules for a merchant."""
    if not merchant_name:
        return None
    
    merchant_key = merchant_name.lower().replace(' ', '').replace(',', '').replace('.', '')
    
    for key, rules in MERCHANT_SPECIFIC_RULES.items():
        if key in merchant_key:
            return rules
    
    return None


# Validation rules
VALIDATION_RULES = {
    'total_must_be_positive': True,
    'tax_cannot_exceed_total': True,
    'subtotal_plus_tax_close_to_total': True,  # Within 5% tolerance
    'date_must_be_reasonable': True,  # Within last 10 years and not future
    'price_reasonable_range': (0.01, 10000.00),  # Min and max reasonable prices
    'merchant_name_not_empty': True,
    'minimum_confidence_score': 0.3
}


def validate_receipt_data(receipt_data: Dict[str, Any]) -> Dict[str, bool]:
    """Validate extracted receipt data against rules."""
    validation_results = {}
    
    # Validate total is positive
    if VALIDATION_RULES['total_must_be_positive']:
        total = receipt_data.get('total')
        validation_results['total_positive'] = total is None or total > 0
    
    # Validate tax doesn't exceed total
    if VALIDATION_RULES['tax_cannot_exceed_total']:
        total = receipt_data.get('total', 0)
        tax = receipt_data.get('tax_amount', 0)
        validation_results['tax_reasonable'] = tax is None or total is None or tax <= total
    
    # Validate subtotal + tax ≈ total
    if VALIDATION_RULES['subtotal_plus_tax_close_to_total']:
        total = receipt_data.get('total')
        subtotal = receipt_data.get('subtotal')
        tax = receipt_data.get('tax_amount')
        
        if all(x is not None for x in [total, subtotal, tax]):
            calculated_total = subtotal + tax
            tolerance = total * 0.05  # 5% tolerance
            validation_results['totals_match'] = abs(total - calculated_total) <= tolerance
        else:
            validation_results['totals_match'] = True  # Can't validate if data missing
    
    return validation_results