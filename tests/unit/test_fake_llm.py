from shop_ops_agent.llm import Message
from tests.fakes import FakeLLM


async def test_fake_llm_records_calls(fake_llm: FakeLLM) -> None:
    resp = await fake_llm.complete([Message(role="user", content="hi")])
    assert resp.content == "ok"
    assert len(fake_llm.calls) == 1


async def test_fake_llm_snapshots_messages(fake_llm: FakeLLM) -> None:
    messages = [Message(role="user", content="hi")]
    await fake_llm.complete(messages)
    messages.append(Message(role="tool", content="{}", tool_call_id="c1"))

    recorded, _ = fake_llm.calls[0]
    assert recorded == [Message(role="user", content="hi")]
