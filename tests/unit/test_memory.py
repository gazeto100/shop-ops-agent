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


def test_new_sessions_get_unguessable_server_issued_ids() -> None:
    memory = SessionMemory()

    first, second = memory.resolve(None), memory.resolve(None)

    assert first != second
    assert len(first) >= 20  # secrets.token_urlsafe(16): 128 random bits


def test_a_client_chosen_id_is_never_adopted() -> None:
    memory = SessionMemory()
    memory.add_exchange(memory.resolve(None), "Аз съм Иван", "Здравейте, Иван!")

    issued = memory.resolve("demo")

    assert issued != "demo"
    assert memory.history(issued) == []


def test_an_issued_id_that_is_still_held_is_kept() -> None:
    memory = SessionMemory()
    session = memory.resolve(None)
    memory.add_exchange(session, "q", "a")

    assert memory.resolve(session) == session


def test_cited_products_of_remembered_answers_are_known() -> None:
    memory = SessionMemory(max_messages=4)
    memory.add_exchange("s", "q1", "a1", {"BOIL-80-A": "Бойлер"})
    memory.add_exchange("s", "q2", "a2", {"KET-17-S": "Кана"})
    assert memory.cited_products("s") == {"BOIL-80-A": "Бойлер", "KET-17-S": "Кана"}

    memory.add_exchange("s", "q3", "a3")  # BOIL-80-A's answer leaves the window

    assert memory.cited_products("s") == {"KET-17-S": "Кана"}
    assert memory.cited_products("unknown") == {}
