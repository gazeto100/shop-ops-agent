import pytest

from shop_ops_agent.llm import Message
from tests.conftest import FakeLLM


@pytest.mark.asyncio
async def test_fake_llm_records_calls(fake_llm: FakeLLM) -> None:
    resp = await fake_llm.complete([Message(role="user", content="hi")])
    assert resp.content == "ok"
    assert len(fake_llm.calls) == 1
