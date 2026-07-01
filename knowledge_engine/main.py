from knowledge_engine.domain.document.document_source import (
    DocumentSource,
    SourceType,
)
from knowledge_engine.loaders.text_loader import TextLoader
from knowledge_engine.parsers.text_parser import TextParser


def main() -> None:
    source = DocumentSource(
        type=SourceType.TEXT,
        name="manuale.txt",
        uri="test_data/manuale.txt",
    )

    loader = TextLoader()
    parser = TextParser()

    raw_document = loader.load(source)

    print("=== RAW DOCUMENT ===")
    print(raw_document)

    document = parser.parse(raw_document)

    print("\n=== DOCUMENT ===")
    print(document)


if __name__ == "__main__":
    main()