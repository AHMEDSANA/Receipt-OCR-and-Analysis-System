# filepath: d:\Hexlar\Reciept File\Paddle_OCR\receipt_analyzer.py
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from receipt_extractor import ReceiptInfo, ReceiptExtractor
import sqlite3


class ReceiptAnalyzer:
    """Advanced receipt analysis and data management."""
    
    def __init__(self, db_path: str = "receipts.db"):
        """Initialize the receipt analyzer with database."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for receipt storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create receipts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                merchant_name TEXT,
                transaction_date TEXT,
                transaction_time TEXT,
                receipt_number TEXT,
                total REAL,
                subtotal REAL,
                tax_amount REAL,
                discount REAL,
                payment_method TEXT,
                receipt_type TEXT,
                confidence_score REAL,
                created_at TEXT,
                raw_text TEXT,
                json_data TEXT
            )
        """)
        
        # Create items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS receipt_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER,
                item_name TEXT,
                price REAL,
                quantity INTEGER,
                category TEXT,
                line_number INTEGER,
                FOREIGN KEY (receipt_id) REFERENCES receipts (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_receipt(self, receipt: ReceiptInfo) -> int:
        """Save receipt to database and return receipt ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert receipt
        cursor.execute("""
            INSERT INTO receipts (
                merchant_name, transaction_date, transaction_time, receipt_number,
                total, subtotal, tax_amount, discount, payment_method, receipt_type,
                confidence_score, created_at, raw_text, json_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            receipt.merchant_name,
            receipt.transaction_date,
            receipt.transaction_time,
            receipt.receipt_number,
            receipt.total,
            receipt.subtotal,
            receipt.tax_amount,
            receipt.discount,
            receipt.payment_method,
            receipt.receipt_type,
            receipt.confidence_score,
            datetime.now().isoformat(),
            receipt.raw_text,
            json.dumps(receipt.__dict__, default=str)
        ))
        
        receipt_id = cursor.lastrowid
        
        # Insert items
        if receipt.items:
            for item in receipt.items:
                cursor.execute("""
                    INSERT INTO receipt_items (
                        receipt_id, item_name, price, quantity, category, line_number
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    receipt_id,
                    item.name,
                    item.price,
                    item.quantity,
                    item.category,
                    item.line_number
                ))
        
        conn.commit()
        conn.close()
        
        return receipt_id
    
    def get_all_receipts(self) -> pd.DataFrame:
        """Get all receipts as a pandas DataFrame."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                id, merchant_name, transaction_date, transaction_time,
                receipt_number, total, subtotal, tax_amount, discount,
                payment_method, receipt_type, confidence_score, created_at
            FROM receipts
            ORDER BY transaction_date DESC, created_at DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_receipt_items(self, receipt_id: int) -> pd.DataFrame:
        """Get items for a specific receipt."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT item_name, price, quantity, category, line_number
            FROM receipt_items
            WHERE receipt_id = ?
            ORDER BY line_number
        """
        
        df = pd.read_sql_query(query, conn, params=(receipt_id,))
        conn.close()
        
        return df
    
    def generate_spending_report(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Generate comprehensive spending report."""
        df = self.get_all_receipts()
        
        if df.empty:
            return {"error": "No receipts found"}
        
        # Filter by date if provided
        if start_date:
            df = df[df['transaction_date'] >= start_date]
        if end_date:
            df = df[df['transaction_date'] <= end_date]
        
        # Calculate metrics
        total_spent = df['total'].sum()
        total_receipts = len(df)
        avg_transaction = df['total'].mean()
        
        # Spending by merchant
        merchant_spending = df.groupby('merchant_name')['total'].agg(['sum', 'count']).reset_index()
        merchant_spending.columns = ['merchant', 'total_spent', 'visit_count']
        merchant_spending = merchant_spending.sort_values('total_spent', ascending=False)
        
        # Spending by category
        category_spending = df.groupby('receipt_type')['total'].agg(['sum', 'count']).reset_index()
        category_spending.columns = ['category', 'total_spent', 'transaction_count']
        category_spending = category_spending.sort_values('total_spent', ascending=False)
        
        # Monthly spending trend
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        monthly_spending = df.groupby(df['transaction_date'].dt.to_period('M'))['total'].sum()
        
        # Payment method analysis
        payment_analysis = df.groupby('payment_method')['total'].agg(['sum', 'count']).reset_index()
        payment_analysis.columns = ['payment_method', 'total_spent', 'usage_count']
        
        return {
            'summary': {
                'total_spent': total_spent,
                'total_receipts': total_receipts,
                'average_transaction': avg_transaction,
                'date_range': {
                    'start': df['transaction_date'].min(),
                    'end': df['transaction_date'].max()
                }
            },
            'merchant_spending': merchant_spending.to_dict('records'),
            'category_spending': category_spending.to_dict('records'),
            'monthly_trend': monthly_spending.to_dict(),
            'payment_analysis': payment_analysis.to_dict('records')
        }
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export all receipts to CSV."""
        df = self.get_all_receipts()
        
        if filename is None:
            filename = f"receipts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(filename, index=False)
        return filename
    
    def export_to_excel(self, filename: str = None) -> str:
        """Export receipts and items to Excel with multiple sheets."""
        if filename is None:
            filename = f"receipts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filename) as writer:
            # Export receipts
            receipts_df = self.get_all_receipts()
            receipts_df.to_excel(writer, sheet_name='Receipts', index=False)
            
            # Export all items
            conn = sqlite3.connect(self.db_path)
            items_query = """
                SELECT 
                    r.merchant_name, r.transaction_date, r.receipt_number,
                    i.item_name, i.price, i.quantity, i.category
                FROM receipt_items i
                JOIN receipts r ON i.receipt_id = r.id
                ORDER BY r.transaction_date DESC, i.line_number
            """
            items_df = pd.read_sql_query(items_query, conn)
            conn.close()
            
            items_df.to_excel(writer, sheet_name='Items', index=False)
            
            # Export summary statistics
            report = self.generate_spending_report()
            if 'error' not in report:
                summary_data = []
                summary_data.append(['Total Spent', report['summary']['total_spent']])
                summary_data.append(['Total Receipts', report['summary']['total_receipts']])
                summary_data.append(['Average Transaction', report['summary']['average_transaction']])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return filename
    
    def create_spending_dashboard(self):
        """Create Streamlit dashboard for spending analysis."""
        st.header("📊 Receipt Analysis Dashboard")
        
        # Load data
        df = self.get_all_receipts()
        
        if df.empty:
            st.warning("No receipts found. Please process some receipts first.")
            return
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime(df['transaction_date']).min())
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime(df['transaction_date']).max())
        
        # Generate report
        report = self.generate_spending_report(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if 'error' in report:
            st.error(report['error'])
            return
        
        # Summary metrics
        st.subheader("📈 Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Spent", f"${report['summary']['total_spent']:.2f}")
        with col2:
            st.metric("Total Receipts", report['summary']['total_receipts'])
        with col3:
            st.metric("Average Transaction", f"${report['summary']['average_transaction']:.2f}")
        with col4:
            avg_per_day = report['summary']['total_spent'] / max((end_date - start_date).days, 1)
            st.metric("Average per Day", f"${avg_per_day:.2f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💳 Spending by Category")
            if report['category_spending']:
                category_df = pd.DataFrame(report['category_spending'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=category_df, x='total_spent', y='category', ax=ax)
                ax.set_xlabel('Total Spent ($)')
                ax.set_title('Spending by Receipt Type')
                st.pyplot(fig)
        
        with col2:
            st.subheader("🏪 Top Merchants")
            if report['merchant_spending']:
                merchant_df = pd.DataFrame(report['merchant_spending'][:10])  # Top 10
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(data=merchant_df, x='total_spent', y='merchant', ax=ax)
                ax.set_xlabel('Total Spent ($)')
                ax.set_title('Top 10 Merchants by Spending')
                st.pyplot(fig)
        
        # Monthly trend
        if len(report['monthly_trend']) > 1:
            st.subheader("📅 Monthly Spending Trend")
            monthly_df = pd.DataFrame(list(report['monthly_trend'].items()), 
                                    columns=['Month', 'Total_Spent'])
            monthly_df['Month'] = pd.to_datetime(monthly_df['Month'].astype(str))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(monthly_df['Month'], monthly_df['Total_Spent'], marker='o')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Spent ($)')
            ax.set_title('Monthly Spending Trend')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        # Data tables
        st.subheader("📋 Detailed Data")
        
        tab1, tab2, tab3 = st.tabs(["Recent Receipts", "Merchant Summary", "Category Summary"])
        
        with tab1:
            filtered_df = df[(pd.to_datetime(df['transaction_date']) >= pd.to_datetime(start_date)) & 
                            (pd.to_datetime(df['transaction_date']) <= pd.to_datetime(end_date))]
            st.dataframe(filtered_df, use_container_width=True)
        
        with tab2:
            if report['merchant_spending']:
                st.dataframe(pd.DataFrame(report['merchant_spending']), use_container_width=True)
        
        with tab3:
            if report['category_spending']:
                st.dataframe(pd.DataFrame(report['category_spending']), use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export to CSV"):
                filename = self.export_to_csv()
                st.success(f"Exported to {filename}")
        
        with col2:
            if st.button("Export to Excel"):
                filename = self.export_to_excel()
                st.success(f"Exported to {filename}")
    
    def search_receipts(self, query: str, search_type: str = "merchant") -> pd.DataFrame:
        """Search receipts by merchant name, item, or other criteria."""
        conn = sqlite3.connect(self.db_path)
        
        if search_type == "merchant":
            sql_query = """
                SELECT * FROM receipts 
                WHERE merchant_name LIKE ? 
                ORDER BY transaction_date DESC
            """
            params = (f"%{query}%",)
        
        elif search_type == "item":
            sql_query = """
                SELECT DISTINCT r.* FROM receipts r
                JOIN receipt_items i ON r.id = i.receipt_id
                WHERE i.item_name LIKE ?
                ORDER BY r.transaction_date DESC
            """
            params = (f"%{query}%",)
        
        elif search_type == "amount":
            try:
                amount = float(query)
                sql_query = """
                    SELECT * FROM receipts 
                    WHERE ABS(total - ?) < 0.01
                    ORDER BY transaction_date DESC
                """
                params = (amount,)
            except ValueError:
                return pd.DataFrame()
        
        else:
            # General text search
            sql_query = """
                SELECT * FROM receipts 
                WHERE raw_text LIKE ? OR merchant_name LIKE ? OR receipt_number LIKE ?
                ORDER BY transaction_date DESC
            """
            params = (f"%{query}%", f"%{query}%", f"%{query}%")
        
        df = pd.read_sql_query(sql_query, conn, params=params)
        conn.close()
        
        return df
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count receipts
        cursor.execute("SELECT COUNT(*) FROM receipts")
        receipt_count = cursor.fetchone()[0]
        
        # Count items
        cursor.execute("SELECT COUNT(*) FROM receipt_items")
        item_count = cursor.fetchone()[0]
        
        # Count merchants
        cursor.execute("SELECT COUNT(DISTINCT merchant_name) FROM receipts WHERE merchant_name IS NOT NULL")
        merchant_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_receipts': receipt_count,
            'total_items': item_count,
            'unique_merchants': merchant_count
        }
    
    def duplicate_detection(self) -> pd.DataFrame:
        """Detect potential duplicate receipts."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                merchant_name, transaction_date, total, 
                COUNT(*) as duplicate_count,
                GROUP_CONCAT(id) as receipt_ids
            FROM receipts 
            WHERE merchant_name IS NOT NULL 
                AND transaction_date IS NOT NULL 
                AND total IS NOT NULL
            GROUP BY merchant_name, transaction_date, total
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df