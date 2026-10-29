from shop_ops_agent.agent.memory import MAX_MESSAGES, SessionMemory
from shop_ops_agent.llm import Message


def test_unknown_session_has_no_history() -> None:
    assert SessionMemory().history("nobody") == []


def test_exchanges_are_kept_in_order_as_text_messages() -> None:
    memory = SessionMemory()

    memory.add_exchange("s1", "Здравей", "Здравейте!")

    assert memory.history("s1") == [
        Message(role="user", content="Здравей"),
        Message(role="assistant", content="Здравейте!"),
    ]


def test_only_the_last_10_messages_are_kept() -> None:
    memory = SessionMemory()
    for i in range(8):
        memory.add_exchange("s1", f"q{i}", f"a{i}")

    history = memory.history("s1")

    assert MAX_MESSAGES == 10
    assert [m.content for m in history] == [
        "q3",
        "a3",
        "q4",
        "a4",
        "q5",
        "a5",
        "q6",
        "a6",
        "q7",
        "a7",
    ]
    assert history[0].role == "user"  # the window always starts on a question


def test_history_is_a_copy() -> None:
    memory = SessionMemory()
    memory.add_exchange("s1", "q", "a")

    memory.history("s1").clear()

    assert len(memory.history("s1")) == 2


def test_least_recently_used_session_is_dropped_beyond_the_cap() -> None:
    memory = SessionMemory(max_sessions=2)
    memory.add_exchange("old", "q", "a")
    memory.add_exchange("busy", "q", "a")
    memory.history("old")  # reading counts as use: "busy" is now the least recent

    memory.add_exchange("new", "q", "a")

    assert memory.history("busy") == []
    assert memory.history("old") != []
    assert memory.history("new") != []
