"""One chat turn: retrieve -> prompt (+ session history) -> tool loop -> guarded answer.

Guardrails run on the final answer, using this turn's tool results as the evidence: a
price or stock figure must appear in a tool result, and a cited product must have come
from retrieval or a tool. Anything else is replaced by a safe reply.
"""

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Literal

from shop_ops_agent.agent.loop import MAX_ITERATIONS, AgentLoop, ToolExchange
from shop_ops_agent.agent.memory import SessionMemory
from shop_ops_agent.agent.prompts import (
    MAX_ITERATIONS_REPLY,
    PRICE_UNAVAILABLE_REPLY,
    PRODUCT_NOT_FOUND_REPLY,
    PROMPT_VERSION,
    REFUSAL_REPLY,
    SYSTEM_PROMPT,
    Language,
    user_prompt,
)
from shop_ops_agent.agent.tools import Audience, ToolRegistry
from shop_ops_agent.llm import LLM, LLMRefusalError, Message, ToolCall
from shop_ops_agent.rag.retriever import Searcher

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
# What a figure in a tool result means, by field name. Only these back an answer.
type FigureKind = Literal["eur", "bgn", "stock"]
_FIGURE_FIELDS: dict[str, FigureKind] = {
    "price_eur": "eur",
    "price_bgn": "bgn",
    "supplier_price_bgn": "bgn",
    "stock_qty": "stock",
}
_NUM = r"(\d(?:[\d\s.,]*\d)?)"
# Price/stock mentions in the answer, by kind. A mention must equal a figure of the SAME
# kind from this turn's tool results: "4 EUR" is not backed by a stock count of 4, and
# "189.00 лв" is not backed by a EUR price of 189.00. Kind None is never backed: no tool
# returns dollars or cent amounts.
_MENTION_PATTERNS: tuple[tuple[FigureKind | None, str], ...] = (
    ("eur", rf"(?:€|\beur\b)\s*{_NUM}|{_NUM}\s*(?:€|\beur\b|евро)"),
    ("bgn", rf"(?:\bbgn\b|\bлв\b\.?)\s*{_NUM}|{_NUM}\s*(?:\bbgn\b|\bлв\b|лева)"),
    ("stock", rf"{_NUM}\s*(?:\bбр\b|броя|\bpcs\b|\bpieces\b|\bunits?\b|\bin stock\b|на склад)"),
    (None, rf"\$\s*{_NUM}|{_NUM}\s*(?:\$|стотинки|\bcents?\b)"),
)
_MENTIONS = tuple((kind, re.compile(p, re.IGNORECASE)) for kind, p in _MENTION_PATTERNS)
_NUMERIC_STRING = re.compile(r"-?\d+(?:\.\d+)?")

type Guardrail = Literal["price_or_stock", "unverified_product", "max_iterations"]


@dataclass(frozen=True)
class CitedProduct:
    sku: str
    name: str


@dataclass(frozen=True)
class ChatResult:
    answer: str
    citations: tuple[CitedProduct, ...]
    tool_calls: tuple[ToolCall, ...]
    session_id: str = ""  # filled in by ChatService.answer: pass it with the next message
    prompt_version: str = PROMPT_VERSION
    # Set when the model's answer was replaced by a safe reply, and why.
    guardrail: Guardrail | None = None
    unverified_skus: tuple[str, ...] = ()


@dataclass
class Evidence:
    """What this turn's tool results establish: products (code -> name) and figures by
    kind (EUR price, BGN price, stock count)."""

    products: dict[str, str] = field(default_factory=dict)
    figures: dict[FigureKind, set[Decimal]] = field(
        default_factory=lambda: {"eur": set(), "bgn": set(), "stock": set()}
    )

    @classmethod
    def from_exchanges(cls, exchanges: tuple[ToolExchange, ...]) -> "Evidence":
        evidence = cls()
        for exchange in exchanges:
            evidence._collect(json.loads(exchange.result_json))
        return evidence

    def _collect(self, node: object) -> None:
        if isinstance(node, dict):
            sku, name = node.get("sku"), node.get("name")
            if isinstance(sku, str) and isinstance(name, str):
                self.products[sku] = name
            for key, value in node.items():
                if (kind := _FIGURE_FIELDS.get(key)) and (figure := _as_figure(value)) is not None:
                    self.figures[kind].add(figure)
                self._collect(value)
        elif isinstance(node, list):
            for value in node:
                self._collect(value)


def _as_figure(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None  # bool is a subclass of int; True is not the figure 1
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str) and _NUMERIC_STRING.fullmatch(value):
        return Decimal(value)  # money is serialised as "189.00"
    return None


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


def parse_figure(text: str) -> Decimal:
    """ "1 299,50" / "1,299.50" / "189.00" -> Decimal. A comma is the decimal separator
    unless a dot is also present (then it groups thousands), as in Bulgarian usage."""
    digits = re.sub(r"\s", "", text)
    digits = digits.replace(",", "") if "." in digits else digits.replace(",", ".")
    return Decimal(digits)


def unbacked_figures(answer: str, evidence: Evidence) -> list[str]:
    """Price/stock mentions in the answer that no tool result of the same kind backs."""
    unbacked = []
    for kind, pattern in _MENTIONS:
        for m in pattern.finditer(answer):
            number = next(g for g in m.groups() if g)
            if kind is None or parse_figure(number) not in evidence.figures[kind]:
                unbacked.append(m.group().strip())
    return unbacked


class ChatService:
    def __init__(
        self,
        retriever: Searcher,
        llm: LLM,
        registry: ToolRegistry,
        memory: SessionMemory,
        *,
        audience: Audience = "customer",
        k: int = 5,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._retriever = retriever
        self._memory = memory
        self._loop = AgentLoop(llm, registry, audience, max_iterations)
        self._k = k

    async def answer(self, session_id: str | None, question: str) -> ChatResult:
        """`session_id` is the id a previous answer returned, or None to start a session.
        An id this service didn't issue starts a new session too."""
        session_id = self._memory.resolve(session_id)
        result = await self._answer(session_id, question)
        # Remember what the user actually saw, guarded replies included, and which
        # products it verifiably cited.
        cited = {c.sku: c.name for c in result.citations}
        self._memory.add_exchange(session_id, question, result.answer, cited)
        return replace(result, session_id=session_id)

    async def _answer(self, session_id: str, question: str) -> ChatResult:
        language = detect_language(question)
        # search() embeds the query on the CPU; run it off the event loop.
        hits = await asyncio.to_thread(self._retriever.search, question, self._k)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT),
            *self._memory.history(session_id),
            Message(role="user", content=user_prompt(question, hits, language)),
        ]
        try:
            outcome = await self._loop.run(messages)
        except LLMRefusalError:
            return ChatResult(answer=REFUSAL_REPLY[language], citations=(), tool_calls=())
        calls = tuple(e.call for e in outcome.exchanges)
        if outcome.final is None:
            return ChatResult(
                answer=MAX_ITERATIONS_REPLY[language],
                citations=(),
                tool_calls=calls,
                guardrail="max_iterations",
            )

        answer = outcome.final.content
        evidence = Evidence.from_exchanges(outcome.exchanges)
        # The prompt forbids unbacked figures, but a prompt is not a guarantee.
        if unbacked := unbacked_figures(answer, evidence):
            log.warning("guardrail price_or_stock %s (prompt %s)", unbacked, PROMPT_VERSION)
            return ChatResult(
                answer=PRICE_UNAVAILABLE_REPLY[language],
                citations=(),
                tool_calls=calls,
                guardrail="price_or_stock",
            )
        # Verified products: cited earlier in this session, retrieved now, or returned by a
        # tool now. Earlier citations make follow-ups ("and its warranty?") work.
        known: Mapping[str, str] = (
            self._memory.cited_products(session_id)
            | {hit.sku: hit.name for hit in hits}
            | evidence.products
        )
        unverified = tuple(s for s in cited_skus(answer) if s not in known)
        if unverified:
            log.warning("guardrail unverified_product %s (prompt %s)", unverified, PROMPT_VERSION)
            return ChatResult(
                answer=PRODUCT_NOT_FOUND_REPLY[language],
                citations=(),
                tool_calls=calls,
                guardrail="unverified_product",
                unverified_skus=unverified,
            )
        return ChatResult(
            answer=answer,
            citations=tuple(CitedProduct(s, known[s]) for s in cited_skus(answer)),
            tool_calls=calls,
        )
