from slugger import slugify


def test_slugify_removes_punctuation_and_collapses_spaces() -> None:
    assert slugify("  Hello,   Local Agent!  ") == "hello-local-agent"


def test_slugify_preserves_word_separators() -> None:
    assert slugify("Bubblewrap sandbox") == "bubblewrap-sandbox"
