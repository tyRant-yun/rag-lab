from __future__ import annotations

import unicodedata
from collections.abc import Iterable

import jieba


class LexicalAnalyzer:
    """Convert source text into BM25-ready lexical terms."""

    def __init__(
        self,
        *,
        stopwords: Iterable[str] = (),
        user_words: Iterable[str] = (),
    ) -> None:
        self._tokenizer = jieba.Tokenizer()

        normalized_stopwords = (
            self._normalize_text(word).strip()
            for word in stopwords
        )
        self._stopwords = frozenset(
            word
            for word in normalized_stopwords
            if word
        )

        for word in user_words:
            normalized_word = self._normalize_text(
                word
            ).strip()

            if not normalized_word:
                raise ValueError(
                    "user words cannot contain empty entries"
                )

            self._tokenizer.add_word(normalized_word)

    def analyze(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized_text = self._normalize_text(text)

        if not normalized_text.strip():
            return ()

        tokens: list[str] = []

        raw_tokens = tuple(
            self._tokenizer.cut(
                normalized_text,
                HMM=False,
            )
        )

        for raw_token in self._merge_dotted_numbers(
            raw_tokens
        ):
            token = raw_token.strip()

            if not token:
                continue

            if not self._is_searchable(token):
                continue

            if token in self._stopwords:
                continue

            tokens.append(token)

        return tuple(tokens)

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = unicodedata.normalize(
            "NFKC",
            text,
        )
        return normalized.casefold()

    @staticmethod
    def _merge_dotted_numbers(
        tokens: tuple[str, ...],
    ) -> tuple[str, ...]:
        merged: list[str] = []
        index = 0

        while index < len(tokens):
            token = tokens[index]

            if token.isdecimal():
                parts = [token]
                cursor = index

                while (
                    cursor + 2 < len(tokens)
                    and tokens[cursor + 1] == "."
                    and tokens[cursor + 2].isdecimal()
                ):
                    parts.extend(
                        (
                            ".",
                            tokens[cursor + 2],
                        )
                    )
                    cursor += 2

                if cursor != index:
                    merged.append("".join(parts))
                    index = cursor + 1
                    continue

            merged.append(token)
            index += 1

        return tuple(merged)

    @staticmethod
    def _is_searchable(token: str) -> bool:
        return any(
            character.isalnum()
            for character in token
        )
