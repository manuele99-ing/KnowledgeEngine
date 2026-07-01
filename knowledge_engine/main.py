from knowledge_engine.domain.block import Block, BlockType
from knowledge_engine.domain.section import Section, SectionType


def main():

    section = Section(
        title="Installazione",
        type=SectionType.CHAPTER,
        blocks=(
            Block(
                type=BlockType.TEXT,
                content="Collegare l'alimentazione."
            ),
        ),
    )

    print(section)


if __name__ == "__main__":
    main()