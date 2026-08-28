"""
Utility functions for downloading data
"""

from .fetch_test_data import fetch_test_data

# create fetch class to group fetching functions
fetch = type("fetch", (), {})
fetch.test_data = fetch_test_data
