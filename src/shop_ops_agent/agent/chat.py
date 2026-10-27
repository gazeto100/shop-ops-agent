"""One grounded chat turn: retrieve -> prompt -> LLM -> answer with verified citations.

No tools and no session memory yet: tools land in #6, the tool loop and memory in #7.
"""

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from shop_ops_agent.agent.prompts import REFUSAL_REPLY, SYSTEM_PROMPT, Language, user_prompt
from shop_ops_agent.llm import LLM, LLMRefusalError, Message, ToolCall
from shop_ops_agent.rag.retriever import Hit

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")
# "[BOIL-80-A: Бойлер …]" -> "BOIL-80-A"
_CITED_SKU = re.compile(r"\[([^\[\]:]+):")


class Searcher(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: tuple[Hit, ...]
    tool_calls: tuple[ToolCall, ...]


def detect_language(text: str) -> Language:
    """English only when the user writes Latin script and no Cyrillic; Bulgarian otherwise
    (the shop's default, also for input with no letters at all)."""
    return "en" if _LATIN.search(text) and not _CYRILLIC.search(text) else "bg"


def cited_hits(answer: str, hits: Sequence[Hit]) -> tuple[Hit, ...]:
    """Retrieved hits the answer actually cites, in order of first citation. A cited sku
    that wasn't retrieved (i.e. invented) is dropped."""
    by_sku = {hit.sku: hit for hit in hits}
    cited = dict.fromkeys(m.strip() for m in _CITED_SKU.findall(answer))
    return tuple(by_sku[sku] for sku in cited if sku in by_sku)


class ChatService:
    def __init__(self, retriever: Searcher, llm: LLM, k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k

    async def answer(self, question: str) -> ChatResult:
        language = detect_language(question)
        # search() embeds the query on the CPU; run it off the event loop.
        hits = await asyncio.to_thread(self._retriever.search, question, self._k)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_prompt(question, hits, language)),
        ]
        try:
            response = await self._llm.complete(messages)
        except LLMRefusalError:
            return ChatResult(answer=REFUSAL_REPLY[language], citations=(), tool_calls=())
        return ChatResult(
            answer=response.content,
            citations=cited_hits(response.content, hits),
            tool_calls=response.tool_calls,
        )
