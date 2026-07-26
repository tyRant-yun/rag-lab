import pytest

from rag_lab.retrieval.lexical import (
    LexicalAnalyzer,
)


def test_empty_text_returns_empty_tuple():
    analyzer = LexicalAnalyzer()

    assert analyzer.analyze("") == ()
    assert analyzer.analyze(" \n\t ") == ()


def test_normalizes_full_width_and_case():
    analyzer = LexicalAnalyzer()

    assert analyzer.analyze("ＴＣＰ") == ("tcp",)


def test_removes_punctuation():
    analyzer = LexicalAnalyzer()

    tokens = analyzer.analyze("TCP，HTTP。")

    assert "tcp" in tokens
    assert "http" in tokens
    assert "，" not in tokens
    assert "。" not in tokens


def test_preserves_term_frequency():
    analyzer = LexicalAnalyzer()

    tokens = analyzer.analyze("RTT RTT")

    assert tokens.count("rtt") == 2


def test_returns_tuple():
    analyzer = LexicalAnalyzer()

    result = analyzer.analyze("计算机网络")

    assert isinstance(result, tuple)


def test_rejects_non_string_input():
    analyzer = LexicalAnalyzer()

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        analyzer.analyze(123)  # type: ignore[arg-type]

def test_supports_domain_user_words():
    analyzer = LexicalAnalyzer(
        user_words=("端到端可靠传输",),
    )

    tokens = analyzer.analyze(
        "端到端可靠传输是运输层的重要能力"
    )

    assert "端到端可靠传输" in tokens


def test_removes_configured_stopwords():
    analyzer = LexicalAnalyzer(
        stopwords=("的", "是"),
    )

    tokens = analyzer.analyze(
        "网络的核心是协议"
    )

    assert "的" not in tokens
    assert "是" not in tokens


def test_normalizes_stopwords():
    analyzer = LexicalAnalyzer(
        stopwords=("ＴＣＰ",),
    )

    assert analyzer.analyze("TCP") == ()


def test_preserves_technical_terms_and_numbers():
    analyzer = LexicalAnalyzer()

    tokens = analyzer.analyze(
        "HTTP/1.1 使用 IPv4"
    )

    assert "http" in tokens
    assert "1.1" in tokens
    assert "ipv4" in tokens


def test_analysis_is_deterministic():
    analyzer = LexicalAnalyzer(
        user_words=("往返时延",),
    )
    text = "TCP 使用往返时延 RTT"

    first = analyzer.analyze(text)
    second = analyzer.analyze(text)

    assert first == second


def test_rejects_empty_user_word():
    with pytest.raises(
        ValueError,
        match="user words cannot contain empty entries",
    ):
        LexicalAnalyzer(
            user_words=("TCP", " "),
        )
