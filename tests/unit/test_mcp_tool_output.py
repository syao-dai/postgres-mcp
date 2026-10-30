"""Regression tests: tool results must reach MCP clients exactly once.

FastMCP (mcp SDK) derives an outputSchema and structuredContent from typed return annotations
such as ``List[TextContent]``. The client then receives the same text in both ``content`` and
``structuredContent``, and an LLM client that forwards both doubles its context cost. The server
disables structured output; these tests fail if an ``mcp`` upgrade changes that behavior.
"""

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from mcp.shared.memory import create_connected_server_and_client_session


@asynccontextmanager
async def connected_client():
    """In-memory MCP client connected to the real server wiring (tools registered by main()).

    A context manager rather than a fixture: the client's anyio task group must be entered and
    exited in the same task, which async fixtures do not guarantee.
    """
    from postgres_mcp.server import main
    from postgres_mcp.server import mcp

    original_argv = sys.argv
    try:
        sys.argv = ["postgres_mcp", "postgresql://user:password@localhost/db"]
        with (
            patch("postgres_mcp.server.db_connection.pool_connect", AsyncMock()),
            patch("postgres_mcp.server.mcp.run_stdio_async", AsyncMock()),
        ):
            await main()
    finally:
        sys.argv = original_argv

    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        yield session


@pytest.mark.asyncio
async def test_no_tool_declares_output_schema():
    async with connected_client() as client:
        tools = (await client.list_tools()).tools
    assert "execute_sql" in {tool.name for tool in tools}
    with_schema = [tool.name for tool in tools if tool.outputSchema is not None]
    assert with_schema == [], f"tools still declare an outputSchema: {with_schema}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_format", "expected_text"),
    [("csv", "a,b\n1,2"), ("markdown", "| a | b |"), ("list", "[{'a': 1, 'b': 2}]")],
)
async def test_execute_sql_returns_single_text_block(output_format, expected_text):
    driver = SimpleNamespace(execute_query=AsyncMock(return_value=[SimpleNamespace(cells={"a": 1, "b": 2})]))
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=driver)):
        async with connected_client() as client:
            result = await client.call_tool("execute_sql", {"sql": "select 1 a, 2 b", "format": output_format})

    assert result.structuredContent is None
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert expected_text in result.content[0].text


@pytest.mark.asyncio
async def test_execute_sql_error_has_no_structured_content():
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(side_effect=RuntimeError("boom"))):
        async with connected_client() as client:
            result = await client.call_tool("execute_sql", {"sql": "select 1"})

    assert result.structuredContent is None
    assert len(result.content) == 1
    assert result.content[0].text == "Error: boom"
