from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b


@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers together"""
    return a - b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together"""
    return a * b


tools = [add_numbers, subtract_numbers, multiply_numbers]

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        content="You are an AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent", should_continue, {"continue": "tools", "end": END}
)
graph.add_edge("tools", "our_agent")
app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 40 + 12. Add 3+4")]}
print_stream(app.stream(inputs, stream_mode="values"))
