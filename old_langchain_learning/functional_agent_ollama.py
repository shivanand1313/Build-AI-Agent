# ── Step 1: Define tools and model ──────────────────────────────────────────

from langchain.tools import tool
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# ── Step 2: Import Functional API tools ─────────────────────────────────────

from langgraph.graph import add_messages
from langchain.messages import SystemMessage, HumanMessage, ToolCall
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task   # <-- The key new imports

# ── Step 3: Define model node as a @task ────────────────────────────────────

@task
def call_llm(messages: list[BaseMessage]):
    """LLM decides whether to call a tool or not"""
    return model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
        + messages
    )

# ── Step 4: Define tool node as a @task ─────────────────────────────────────

@task
def call_tool(tool_call: ToolCall):
    """Performs the tool call"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

# ── Step 5: Define the agent as an @entrypoint ──────────────────────────────

@entrypoint()
def agent(messages: list[BaseMessage]):

    # First LLM call
    model_response = call_llm(messages).result()

    # Loop: keep calling tools until LLM is done
    while True:
        if not model_response.tool_calls:   # No tool needed → stop
            break

        # Run all tool calls (could be multiple at once)
        tool_result_futures = [call_tool(tc) for tc in model_response.tool_calls]
        tool_results = [fut.result() for fut in tool_result_futures]

        # Add responses to message history and call LLM again
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    # Final answer
    messages = add_messages(messages, model_response)
    return messages

# ── Run it ───────────────────────────────────────────────────────────────────

messages = [HumanMessage(content="Add 3 and 4.")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print("\n")