import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_extractor import TextExtractor


def main():
    extractor = TextExtractor()
    extractor.process_all_pdfs()
    return True


if __name__ == "__main__":
    main()
