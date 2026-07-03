from __future__ import annotations

import re

from knowledge_engine.parsers.context.parse_context import ParseContext
from knowledge_engine.pipeline.pipeline_component import PipelineComponent


class ValidateHeadingsComponent(PipelineComponent):
    """
    Post-processes detected headings to correct false positives.
    
    Validates that implicit headings are actually titles and not narrative content.
    """

    # Words that indicate narrative/descriptive content, not titles
    _narrative_starters = {
        "consultarono", "abbiamo", "questo", "il problema", "durante", "mentre", 
        "quando", "dopo", "prima", "con", "senza", "perché", "anche", "tuttavia", 
        "inoltre", "invece", "comunque", "pur", "nonostante", "realizzando", 
        "scoprendo", "comprendo", "attraverso", "grazie", "sebbene", "benché",
        "malgrado", "fuori", "dentro", "sopra", "sotto", "tra", "presso"
    }
    
    # Pattern for sentences that describe actions (subject + verb pattern)
    _sentence_pattern = re.compile(
        r"^[A-Z][a-z\s]+\s+(ha|è|sono|erano|sarà|saranno|viene|vengono|vennero|stava|stavano|avrebbe|avrebbero|fa|faceva|facevano)\b"
    )

    def process(self, data: ParseContext) -> ParseContext:
        """
        Validate and correct implicit headings.
        """
        corrected_paragraphs = []
        corrected_heading_indexes = set()
        corrected_heading_levels = {}

        for index, paragraph in enumerate(data.paragraphs):
            if paragraph.is_heading and paragraph.heading_style == "implicit":
                # Double-check if this implicit heading is actually valid
                if self._is_content_not_heading(paragraph, data.paragraphs, index):
                    # Mark as content, not heading
                    paragraph = type(paragraph)(
                        text=paragraph.text,
                        original_lines=paragraph.original_lines,
                        start_line=paragraph.start_line,
                        end_line=paragraph.end_line,
                        indentation=paragraph.indentation,
                        blank_lines_before=paragraph.blank_lines_before,
                        blank_lines_after=paragraph.blank_lines_after,
                        heading_level=None,
                        heading_style=None,
                        is_heading=False,
                    )
                else:
                    # Keep it as heading
                    corrected_heading_indexes.add(index)
                    if paragraph.heading_level is not None:
                        corrected_heading_levels[index] = paragraph.heading_level
            elif paragraph.is_heading and paragraph.heading_style in ["markdown", "numeric"]:
                # Markdown and numeric headings are always valid
                corrected_heading_indexes.add(index)
                if paragraph.heading_level is not None:
                    corrected_heading_levels[index] = paragraph.heading_level

            corrected_paragraphs.append(paragraph)

        data.paragraphs = corrected_paragraphs
        data.heading_indexes = corrected_heading_indexes
        data.heading_levels = corrected_heading_levels

        return data

    def _is_content_not_heading(self, paragraph, all_paragraphs: list, index: int) -> bool:
        """
        Determine if an implicit heading is actually narrative content.
        
        Returns True if it should be content, False if it should remain a heading.
        """
        text = paragraph.text.strip()
        text_lower = text.lower()

        # Check 1: Starts with narrative connector
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in self._narrative_starters:
            return True

        # Check 2: Contains sentence pattern (subject + verb)
        if self._sentence_pattern.match(text):
            return True

        # Check 3: Ends with colon AND previous paragraph is a heading
        # This typically means it's introduction to a section, not a sub-heading
        if text.endswith(":") and index > 0:
            prev_para = all_paragraphs[index - 1]
            if prev_para.is_heading and len(text.split()) > 4:
                return True

        # Check 4: Contains typical descriptive phrases
        descriptive_phrases = [
            "ha un potere", "possiede un", "è un", "sono", "era un", "erano",
            "è stato", "è stata", "sono stati", "sono state",
            "secondo me", "io credo", "penso che", "sembra che",
        ]
        if any(phrase in text_lower for phrase in descriptive_phrases):
            return True

        # Check 5: Very long for an implicit heading (> 60 chars)
        if len(text) > 70:
            return True

        # Check 6: Comma in the middle suggests it's a clause, not a title
        if "," in text and len(text.split()) > 6:
            return True

        return False
