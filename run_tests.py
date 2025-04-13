import unittest

import tests.data.test_cleaner as test_cleaner

def main():
    test_cleaner.TestDataCleaner("subTest").test_initialization()

if __name__ == "__main__":
    exit(main())