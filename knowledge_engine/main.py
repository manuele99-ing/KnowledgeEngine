from knowledge_engine.domain.document.document_source import (
    DocumentSource,
    SourceType,
)
from knowledge_engine.loaders.text_loader import TextLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.wiki.markdown_exporter import MarkdownExporter
from knowledge_engine.wiki.wiki_builder import WikiBuilder

# Da riattivare quando testeremo il reasoning
# from knowledge_engine.llm.llm_client_factory import create_llm_client_from_env
# from knowledge_engine.reasoning.reasoning_engine import ReasoningEngine


def main() -> None:
    source = DocumentSource(
        type=SourceType.TEXT,
        name="manuale.txt",
        uri="test_data/manuale.txt",
    )

    loader = TextLoader()
    parser = TextParser()
    extractor = KnowledgeExtractor()

    wiki_builder = WikiBuilder()
    markdown_exporter = MarkdownExporter()

    raw_document = loader.load(source)

    print("=== RAW DOCUMENT ===")
    print(raw_document)

    document = parser.parse(raw_document)

    print("\n=== DOCUMENT ===")
    print(document)

    knowledge_base = extractor.extract(document)

    print("\n=== KNOWLEDGE BASE ===")
    print(knowledge_base)

    wiki = wiki_builder.build(knowledge_base)

    print("\n=== WIKI ===")
    print(wiki)

    markdown_exporter.export(
        wiki=wiki,
        output_directory="output/wiki",
    )

    print("\n✓ Wiki exported to output/wiki")

    # ------------------------------
    # Reasoning test (future)
    # ------------------------------

    # llm_client = create_llm_client_from_env()
    # reasoning_engine = ReasoningEngine(llm_client=llm_client)
    #
    # answer = reasoning_engine.answer(
    #     question="Which protocol does Hub 2 use?",
    #     knowledge_base=knowledge_base,
    # )
    #
    # print("\n=== ANSWER ===")
    # print(answer)


if __name__ == "__main__":
    main()