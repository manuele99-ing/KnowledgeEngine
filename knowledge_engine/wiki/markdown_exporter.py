
from __future__ import annotations

from pathlib import Path
import re

from knowledge_engine.wiki.wiki import Wiki


class MarkdownExporter:
    """
    Exports a Wiki to a Markdown directory structure.
    """

    def export(
        self,
        wiki: Wiki,
        output_directory: str | Path,
    ) -> None:

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        self._write_index(wiki, output_directory)

        pages_directory = output_directory / "pages"
        pages_directory.mkdir(exist_ok=True)

        for page in wiki.pages:
            filename = self._slugify(page.title) + ".md"

            (pages_directory / filename).write_text(
                page.content,
                encoding="utf-8",
            )

    def _write_index(
        self,
        wiki: Wiki,
        output_directory: Path,
    ) -> None:

        lines: list[str] = []

        lines.append(f"# {wiki.index.title}")
        lines.append("")
        lines.append("## Pages")
        lines.append("")

        for page in sorted(
            wiki.pages,
            key=lambda page: page.title.casefold(),
        ):

            filename = self._slugify(page.title)

            lines.append(
                f"- [{page.title}](pages/{filename}.md)"
            )

        (output_directory / "index.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    def _slugify(
        self,
        value: str,
    ) -> str:

        value = value.lower()

        value = re.sub(r"[^\w\s-]", "", value)

        value = re.sub(r"[-\s]+", "_", value)

        return value.strip("_")