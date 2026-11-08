"""Text cleaning utilities for the NewsLens project."""
import html
import re


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_TEXT_PATTERN = re.compile(r"[^a-zA-Z\s]")
SPACE_PATTERN = re.compile(r"\s+")


def clean_article(text: str) -> str:
    """Normalize an article for TF-IDF feature extraction."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = text.lower()
    text = NON_TEXT_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def combine_title_and_body(title: str, body: str) -> str:
    """Combine headline and article body into one model input."""
    return clean_article(f"{title or ''} {body or ''}")
