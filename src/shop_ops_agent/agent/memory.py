"""Session memory (#7): the last N text messages per session, in process memory.

Only text turns are kept (the user's question and the answer they were shown), never
tool calls, tool results or thinking blocks. A sliding window changes the history every
turn, and Claude rejects a replayed thinking block whose preceding history changed, so
those live only inside one turn's tool loop. It also means old prices are never in
context: the model has to call check_stock_and_price again.

Limits of this in-memory version: lost on restart, not shared between workers, and
concurrent requests for one session may interleave. Upgrade path: the same interface
backed by Redis, one list per session -
  RPUSH session:{id} <json>   then   LTRIM session:{id} -N -1   and   EXPIRE session:{id} <ttl>
which bounds each session, shares memory across workers and expires idle sessions
(replacing the LRU cap below).
"""

from collections import OrderedDict, deque

from shop_ops_agent.llm import Message

MAX_MESSAGES = 10  # 5 question/answer exchanges
MAX_SESSIONS = 10_000  # least recently used sessions are dropped beyond this


class SessionMemory:
    def __init__(self, max_messages: int = MAX_MESSAGES, max_sessions: int = MAX_SESSIONS) -> None:
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[Message]] = OrderedDict()

    def history(self, session_id: str) -> list[Message]:
        messages = self._sessions.get(session_id)
        if messages is None:
            return []
        self._sessions.move_to_end(session_id)
        return list(messages)  # a copy: callers can't mutate the stored history

    def add_exchange(self, session_id: str, question: str, answer: str) -> None:
        messages = self._sessions.get(session_id)
        if messages is None:
            messages = self._sessions[session_id] = deque(maxlen=self._max_messages)
        self._sessions.move_to_end(session_id)
        messages.append(Message(role="user", content=question))
        messages.append(Message(role="assistant", content=answer))
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
