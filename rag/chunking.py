"""Structure-aware chunking.

* Text blocks are packed line by line up to `chunk_size`, never cutting a line in
  the middle unless it is longer than the limit; consecutive chunks overlap a little.
* Headings (1. PURPOSE, SECTION TITLES, ...) start a new chunk and travel with it as
  `section` metadata, which makes citations and retrieval more precise.
* Table blocks (CSV/XLSX rows) are packed row by row, without overlap, and every
  chunk repeats the table prefix so it stays self-describing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parsers import Block

_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Z0-9 &,\-/()]{3,70}\s*$")
_NUM_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+[A-Z][\w ,&/\-()]{2,70}$")
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    location: str
    section: str
    index: int = 0


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or s.endswith((".", ",", ";", ":")) or len(s) > 80:
        return False
    if _HEADING.match(s):
        return True
    # "2. Leave Entitlements" style headings: numbered, short, no sentence punctuation
    return bool(_NUM_HEADING.match(s)) and len(s.split()) <= 8


def _split_long(line: str, size: int) -> list[str]:
    if len(line) <= size:
        return [line]
    parts, cur = [], ""
    for sent in _SENT.split(line):
        while len(sent) > size:                     # hard cut as a last resort
            if cur:
                parts.append(cur); cur = ""
            parts.append(sent[:size]); sent = sent[size:]
        if len(cur) + len(sent) + 1 > size and cur:
            parts.append(cur); cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def _overlap_tail(lines: list[str], overlap: int) -> list[str]:
    tail, total = [], 0
    for ln in reversed(lines):
        if total + len(ln) > overlap:
            break
        tail.insert(0, ln); total += len(ln) + 1
    return tail


def chunk_blocks(blocks: list[Block], chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        is_table = block.kind == "table"
        units: list[tuple[str, str]] = []           # (section, line)
        section = ""
        for raw in block.text.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if not is_table and _is_heading(line):
                section = line
            for piece in _split_long(line, chunk_size):
                units.append((section, piece))

        cur: list[str] = []
        cur_len = 0
        cur_section = ""

        def emit():
            body = "\n".join(cur).strip()
            if body:
                text = f"{block.prefix}\n{body}" if block.prefix else body
                chunks.append(Chunk(text, block.location, cur_section))

        for sec, line in units:
            new_section = (not is_table) and cur and sec != cur_section and _is_heading(line)
            if cur and (cur_len + len(line) + 1 > chunk_size or new_section):
                emit()
                carry = [] if (is_table or new_section) else _overlap_tail(cur, overlap)
                cur = carry
                cur_len = sum(len(x) + 1 for x in cur)
            if not cur:
                cur_section = sec
            cur.append(line)
            cur_len += len(line) + 1
        if cur:
            emit()

    for i, c in enumerate(chunks):
        c.index = i
    return chunks
