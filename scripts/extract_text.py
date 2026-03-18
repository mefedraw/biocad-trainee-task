import sys
from pathlib import Path

from src.preprocessing.text_extractor import TextExtractor


def main():
    extractor = TextExtractor()
    extractor.process_all_pdfs()
    return True


if __name__ == "__main__":
    main()
