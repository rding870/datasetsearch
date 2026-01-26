"""Quick script to export database to Google Sheets"""
from database import DatasetDatabase

SPREADSHEET_ID = "1I6xzTmSohoNaQPDd9yDnN8U5WIWRkc3fiS56NTL8BEk"

db = DatasetDatabase("datasets.db")
db.export_to_google_sheets(SPREADSHEET_ID)
