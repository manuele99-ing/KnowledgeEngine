from knowledge_engine.chat.chat_session import ChatSession
from knowledge_engine.loaders.directory_loader import DirectoryLoader
from knowledge_engine.parsers.text_parser import TextParser
from knowledge_engine.reasoning.reasoning_engine import ReasoningEngine
from knowledge_engine.retrieval.config.retrieval_config import RetrievalConfig
from knowledge_engine.retrieval.retrieval_engine import RetrievalEngine
from knowledge_engine.semantic.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.wiki.markdown_exporter import MarkdownExporter
from knowledge_engine.wiki.wiki_builder import WikiBuilder
from knowledge_engine.llm.llm_client_factory import create_llm_client


def main() -> None:

    print("=" * 60)
    print("KnowledgeEngine")
    print("=" * 60)

    print("\nScanning input directory...")

    loader = DirectoryLoader()
    parser = TextParser()
    extractor = KnowledgeExtractor()
    wiki_builder = WikiBuilder()
    markdown_exporter = MarkdownExporter()

    raw_documents = loader.load_directory("input")

    print(f"✓ {len(raw_documents)} document(s) loaded")

    print("\nParsing documents...")

    documents = [
        parser.parse(raw_document)
        for raw_document in raw_documents
    ]

    print(f"✓ {len(documents)} document(s) parsed")

    print("\nExtracting knowledge...")

    knowledge_base = extractor.extract(documents)

    print(f"✓ {len(knowledge_base.entities)} entities")
    print(f"✓ {len(knowledge_base.relations)} relations")

    print("\nGenerating wiki...")

    wiki = wiki_builder.build(knowledge_base)

    markdown_exporter.export(
        wiki=wiki,
        output_directory="output/wiki",
    )

    print("✓ Wiki exported")

    print("\nBuilding retrieval index...")

    retrieval_config = RetrievalConfig(
        debug_enabled=True,
    )

    retrieval_engine = RetrievalEngine(
        knowledge_base=knowledge_base,
        config=retrieval_config,
    )

    print("✓ Retrieval index ready")

    print("\nInitializing LLM...")

    llm_client = create_llm_client()

    reasoning_engine = ReasoningEngine(
        llm_client=llm_client,
        retrieval_engine=retrieval_engine,
    )

    print("✓ LLM ready")

    chat = ChatSession(
        reasoning_engine=reasoning_engine,
    )

    chat.run()


if __name__ == "__main__":
    main()