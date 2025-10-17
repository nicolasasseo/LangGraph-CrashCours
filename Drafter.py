from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Update the document content"""
    global document_content
    document_content = content
    return f"Document content updated; The current content is: \n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the document to a text file and finish the process.
    Args:
      filename: The name of the text file.
    """
    global document_content
    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as f:
            f.write(document_content)
        return f"Document saved to {filename}"
    except Exception as e:
        return f"Error saving document: {e}"


tools = [update, save]

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """
    )

    user_input = input("\n What would you like to do with the document?")
    user_message = HumanMessage(content=user_input)
    messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(messages)
    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"Using tools: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": [user_message, response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    if not messages:
        return "continue"
    else:
        for message in reversed(messages):
            if (
                (isinstance(message, ToolMessage))
                and "saved" in message.content.lower()
                and "document" in message.content.lower()
            ):
                return "end"
        return "continue"


def print_messages(messages):
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"Tool result: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools))
graph.add_edge("our_agent", "tools")
graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "tools", should_continue, {"continue": "our_agent", "end": END}
)
app = graph.compile()


def run_agent():
    print("\n ==========DRAFTER ================ \n")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ==========DRAFTER COMPLETED ================ \n")


if __name__ == "__main__":
    run_agent()
