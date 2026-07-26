import pytest
from pydantic import ValidationError

from rag_lab.contracts import SearchFilters


def test_search_filters_are_publicly_exported():
    import rag_lab.contracts as contracts

    assert "SearchFilters" in contracts.__all__


def test_empty_search_filters_are_valid():
    filters = SearchFilters()

    assert filters.to_dict() == {
        "document_ids": None,
        "heading_prefix": None,
        "page_start": None,
        "page_end": None,
    }


def test_search_filters_serialize_contract():
    filters = SearchFilters(
        document_ids=[
            "sha256:document-a",
            "sha256:document-b",
        ],
        heading_prefix=[
            "第1章 计算机网络和因特网",
            "1.1 什么是因特网",
        ],
        page_start=19,
        page_end=23,
    )

    assert filters.to_dict() == {
        "document_ids": [
            "sha256:document-a",
            "sha256:document-b",
        ],
        "heading_prefix": [
            "第1章 计算机网络和因特网",
            "1.1 什么是因特网",
        ],
        "page_start": 19,
        "page_end": 23,
    }


def test_document_ids_cannot_be_empty():
    with pytest.raises(
        ValidationError,
        match="document_ids cannot be empty",
    ):
        SearchFilters(document_ids=[])


def test_document_id_entries_cannot_be_empty():
    with pytest.raises(
        ValidationError,
        match="document_ids entries cannot be empty",
    ):
        SearchFilters(
            document_ids=["sha256:document", " "]
        )


def test_document_ids_cannot_contain_duplicates():
    with pytest.raises(
        ValidationError,
        match="document_ids cannot contain duplicates",
    ):
        SearchFilters(
            document_ids=[
                "sha256:document",
                "sha256:document",
            ]
        )


def test_heading_prefix_cannot_be_empty():
    with pytest.raises(
        ValidationError,
        match="heading_prefix cannot be empty",
    ):
        SearchFilters(heading_prefix=[])


def test_heading_entries_cannot_be_empty():
    with pytest.raises(
        ValidationError,
        match="heading_prefix entries cannot be empty",
    ):
        SearchFilters(
            heading_prefix=["第一章", " "]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("page_start", 0),
        ("page_end", 0),
    ],
)
def test_page_filters_must_be_positive(
    field_name: str,
    value: int,
):
    with pytest.raises(
        ValidationError,
        match=(
            f"{field_name} must be at least 1"
        ),
    ):
        SearchFilters(**{field_name: value})


def test_page_end_cannot_precede_start():
    with pytest.raises(
        ValidationError,
        match="page_end must not precede page_start",
    ):
        SearchFilters(
            page_start=23,
            page_end=19,
        )


def test_page_filters_are_strict():
    with pytest.raises(ValidationError):
        SearchFilters(page_start="19")


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SearchFilters.model_validate(
            {
                "page_start": 19,
                "unknown_filter": "value",
            },
            strict=True,
        )
