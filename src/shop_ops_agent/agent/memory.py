"""Session memory (#7): the last N text messages per session, in process memory.

Only text turns are kept (the user's question and the answer they were shown), never
tool calls, tool results or thinking blocks. A sliding window changes the history every
turn, and Claude rejects a replayed thinking block whose preceding history changed, so
those live only inside one turn's tool loop. It also means old prices are never in
context: the model has to call check_stock_and_price again. What is kept with each
answer is the list of products it verifiably cited, so a follow-up ("and its
warranty?") may cite them again.

Session ids are issued here, never chosen by the client: an unknown or expired id gets a
fresh random one, so nobody can read another session's history by guessing or reusing
an id such as "demo".

Limits of this in-memory version: lost on restart, not shared between workers, and
concurrent requests for one session may interleave. Upgrade path: the same interface
backed by Redis, one list per session -
  RPUSH session:{id} <json>   then   LTRIM session:{id} -N -1   and   EXPIRE session:{id} <ttl>
which bounds each session, shares memory across workers and expires idle sessions
(replacing the LRU cap below).
"""

import secrets
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass

from shop_ops_agent.llm import Message

MAX_MESSAGES = 10  # 5 question/answer exchanges
MAX_SESSIONS = 10_000  # least recently used sessions are dropped beyond this


@dataclass(frozen=True)
class _Entry:
    message: Message
    cited: tuple[tuple[str, str], ...] = ()  # (sku, name) verified in this answer


class SessionMemory:
    def __init__(self, max_messages: int = MAX_MESSAGES, max_sessions: int = MAX_SESSIONS) -> None:
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[_Entry]] = OrderedDict()

    def resolve(self, session_id: str | None) -> str:
        """The id to use for this request: the given one if this memory issued it and
        still holds it, otherwise a new unguessable one (returned to the client)."""
        if session_id is not None and session_id in self._sessions:
            return session_id
        return secrets.token_urlsafe(16)

    def history(self, session_id: str) -> list[Message]:
        entries = self._sessions.get(session_id)
        if entries is None:
            return []
        self._sessions.move_to_end(session_id)
        return [e.message for e in entries]  # a copy: callers can't mutate the history

    def cited_products(self, session_id: str) -> dict[str, str]:
        """Products (sku -> name) verifiably cited by answers still in the window."""
        entries = self._sessions.get(session_id, ())
        return {sku: name for e in entries for sku, name in e.cited}

    def add_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        cited: Mapping[str, str] | None = None,
    ) -> None:
        entries = self._sessions.get(session_id)
        if entries is None:
            entries = self._sessions[session_id] = deque(maxlen=self._max_messages)
        self._sessions.move_to_end(session_id)
        entries.append(_Entry(Message(role="user", content=question)))
        entries.append(
            _Entry(Message(role="assistant", content=answer), tuple((cited or {}).items()))
        )
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
