from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkMetrics:
    """
    Collects benchmark statistics.
    """

    total_questions: int = 0

    #
    # Retrieval
    #

    correct_documents: int = 0
    correct_sections: int = 0

    #
    # Reasoning
    #

    reasoning_success: int = 0

    #
    # Keywords
    #

    keyword_hits: int = 0
    keyword_total: int = 0

    #
    # Hallucinations
    #

    hallucination_failures: int = 0

    #
    # Negative questions
    #

    negative_questions: int = 0
    negative_success: int = 0

    #
    # Performance
    #

    total_search_time_ms: float = 0.0
    total_answer_time_ms: float = 0.0

    ###########################################################################

    def add_retrieval(
        self,
        *,
        document_ok: bool,
        section_ok: bool,
    ) -> None:

        if document_ok:
            self.correct_documents += 1

        if section_ok:
            self.correct_sections += 1

    ###########################################################################

    def add_reasoning(
        self,
        success: bool,
    ) -> None:

        if success:
            self.reasoning_success += 1

    ###########################################################################

    def add_keyword_score(
        self,
        *,
        found: int,
        expected: int,
    ) -> None:

        self.keyword_hits += found
        self.keyword_total += expected

    ###########################################################################

    def add_hallucination(
        self,
        hallucinated: bool,
    ) -> None:

        if hallucinated:
            self.hallucination_failures += 1

    ###########################################################################

    def add_negative(
        self,
        success: bool,
    ) -> None:

        self.negative_questions += 1

        if success:
            self.negative_success += 1

    ###########################################################################

    def add_search_time(
        self,
        milliseconds: float,
    ) -> None:

        self.total_search_time_ms += milliseconds

    ###########################################################################

    def add_answer_time(
        self,
        milliseconds: float,
    ) -> None:

        self.total_answer_time_ms += milliseconds

    ###########################################################################

    @staticmethod
    def _percentage(
        value: int,
        total: int,
    ) -> float:

        if total == 0:
            return 0.0

        return round(
            value / total * 100,
            2,
        )

    ###########################################################################
    # Scores
    ###########################################################################

    @property
    def document_score(self) -> float:

        return self._percentage(
            self.correct_documents,
            self.total_questions,
        )

    @property
    def section_score(self) -> float:

        return self._percentage(
            self.correct_sections,
            self.total_questions,
        )

    @property
    def reasoning_score(self) -> float:

        return self._percentage(
            self.reasoning_success,
            self.total_questions,
        )

    @property
    def keyword_score(self) -> float:

        return self._percentage(
            self.keyword_hits,
            self.keyword_total,
        )

    @property
    def hallucination_score(self) -> float:

        if self.total_questions == 0:
            return 100.0

        return round(
            (
                1
                - self.hallucination_failures
                / self.total_questions
            )
            * 100,
            2,
        )

    @property
    def negative_score(self) -> float:

        return self._percentage(
            self.negative_success,
            self.negative_questions,
        )

    @property
    def average_search_time_ms(self) -> float:

        if self.total_questions == 0:
            return 0.0

        return round(
            self.total_search_time_ms
            / self.total_questions,
            2,
        )

    @property
    def average_answer_time_ms(self) -> float:

        if self.total_questions == 0:
            return 0.0

        return round(
            self.total_answer_time_ms
            / self.total_questions,
            2,
        )

    @property
    def overall_score(self) -> float:

        score = (
            self.document_score * 0.20
            + self.section_score * 0.20
            + self.reasoning_score * 0.20
            + self.keyword_score * 0.15
            + self.hallucination_score * 0.10
            + self.negative_score * 0.15
        )

        return round(score, 2)