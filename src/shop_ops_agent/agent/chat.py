"""One grounded chat turn: retrieve -> prompt -> LLM -> guarded answer with verified citations.

No tools and no session memory yet: tools land in #6, the tool loop and memory in #7.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from shop_ops_agent.agent.prompts import (
    PRICE_UNAVAILABLE_REPLY,
    PRODUCT_NOT_FOUND_REPLY,
    PROMPT_VERSION,
    REFUSAL_REPLY,
    SYSTEM_PROMPT,
    Language,
    user_prompt,
)
from shop_ops_agent.llm import LLM, LLMRefusalError, Message, ToolCall
from shop_ops_agent.rag.retriever import Hit, Searcher

log = logging.getLogger(__name__)

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Unambiguous English words. Short words that are also transliterated Bulgarian
# ("do" = до, "i" = и, "a" = а, "to" = то, "me" = ме) are deliberately left out.
_ENGLISH_WORD = re.compile(
    r"\b(?:the|is|are|you|your|what|which|how|have|has|does|can|could|would|should|will"
    r"|there|this|that|these|those|need|want|buy|sell|price|cost|much|many|any|with|for"
    r"|and|of|in|it|my|please|hello|hi|thanks|thank)\b",
    re.IGNORECASE,
)
# "[BOIL-80-A: …]" and the "[sku: BOIL-80-A: …]" form models copy from instructions.
_CITATION = re.compile(r"\[\s*(?:(?i:sku)\s*:\s*)?([A-Za-z0-9][A-Za-z0-9._/-]*)\s*:")
# A money amount or a stock count. Nothing in this turn can supply one (no tools until
# #6), so any match is invented. From #6 on, allow only figures that appear verbatim in
# the turn's tool results.
_PRICE_OR_STOCK = re.compile(
    r"(?:€|\$|\beur\b|\bbgn\b|\bлв\b\.?)\s*\d"
    r"|\d[\d\s.,]*\s*(?:€|\$|\beur\b|\bbgn\b|евро|\bлв\b|лева|стотинки|\bcents?\b)"
    r"|\d+\s*(?:\bбр\b|броя|\bpcs\b|\bpieces\b|\bunits?\b|\bin stock\b|на склад)",
    re.IGNORECASE,
)

type Guardrail = Literal["price_or_stock", "unverified_product"]


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: tuple[Hit, ...]
    tool_calls: tuple[ToolCall, ...]
    prompt_version: str = PROMPT_VERSION
    # Set when the model's answer was replaced by a safe reply, and why.
    guardrail: Guardrail | None = None
    unverified_skus: tuple[str, ...] = ()


def detect_language(text: str) -> Language:
    """English only for Latin script with real English words and no Cyrillic. Bulgarian
    otherwise: the shop's default, also for Bulgarian typed in Latin letters and for
    input with no words at all."""
    if _CYRILLIC.search(text) or not _ENGLISH_WORD.search(text):
        return "bg"
    return "en"


def cited_skus(answer: str) -> tuple[str, ...]:
    """Product codes the answer cites, in order of first citation, without duplicates.
    Codes are ASCII and contain a digit or hyphen, so "[Note: …]" is not a citation."""
    found = (m for m in _CITATION.findall(answer) if any(c.isdigit() or c == "-" for c in m))
    return tuple(dict.fromkeys(found))


def cited_hits(answer: str, hits: Sequence[Hit]) -> tuple[Hit, ...]:
    """Retrieved hits the answer actually cites, in order of first citation."""
    by_sku = {hit.sku: hit for hit in hits}
    return tuple(by_sku[sku] for sku in cited_skus(answer) if sku in by_sku)


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

        # The prompt forbids these, but a prompt is not a guarantee: enforce in code.
        if _PRICE_OR_STOCK.search(response.content):
            log.warning("guardrail price_or_stock (prompt %s)", PROMPT_VERSION)
            return ChatResult(
                answer=PRICE_UNAVAILABLE_REPLY[language],
                citations=(),
                tool_calls=response.tool_calls,
                guardrail="price_or_stock",
            )
        retrieved = {hit.sku for hit in hits}
        unverified = tuple(s for s in cited_skus(response.content) if s not in retrieved)
        if unverified:
            log.warning("guardrail unverified_product %s (prompt %s)", unverified, PROMPT_VERSION)
            return ChatResult(
                answer=PRODUCT_NOT_FOUND_REPLY[language],
                citations=(),
                tool_calls=response.tool_calls,
                guardrail="unverified_product",
                unverified_skus=unverified,
            )
        return ChatResult(
            answer=response.content,
            citations=cited_hits(response.content, hits),
            tool_calls=response.tool_calls,
        )
