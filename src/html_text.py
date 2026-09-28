"""HTML 원문을 변경하지 않고 검색용 텍스트만 파생합니다."""

from __future__ import annotations

import re
from html.parser import HTMLParser

BLOCK_TAGS = frozenset((
    "address", "article", "aside", "blockquote", "caption", "div", "dl", "dt", "dd",
    "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4",
    "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "tbody", "thead", "tfoot", "tr", "ul",
))
SKIP_TAGS = frozenset(("script", "style", "noscript", "template"))


class SearchTextParser(HTMLParser):
    """표의 셀 구분과 이미지 대체문구를 살린 검색 전용 파서입니다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.has_table = False
        self.has_image = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag == "table":
            self.has_table = True
        if tag == "img":
            self.has_image = True
            attributes = dict(attrs)
            alt = (attributes.get("alt") or "").strip()
            if alt:
                self.parts.extend((" ", alt, " "))
        elif tag == "br" or tag in BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


def html_to_search_text(html: str) -> tuple[str, bool, bool]:
    """HTML에서 검색용 텍스트와 표·이미지 존재 여부를 반환합니다."""

    parser = SearchTextParser()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts).replace("\u00a0", " ")
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r"[\t \f\v]+", " ", line).strip()
        if normalized:
            lines.append(normalized)
    return "\n".join(lines), parser.has_table, parser.has_image
