from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import Literal
from typing_extensions import Annotated, TypedDict
from src.provider import *
import operator

model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

tools = [FinancialDataProvider.get_asset_data, FinancialDataProvider.get_multiple_assets]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def llm_call(state: MessageState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with researching financial metrics"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def tool_node(state: MessageState):
    """Performs the tool call"""

    result = []
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            
    return {"messages": result}

def should_continue(state: MessageState) -> Literal["tool_node", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "tool_node"

    return "__end__"

# Build workflow
agent_builder = StateGraph(MessageState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connected nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    source="llm_call",
    path=should_continue,
    path_map=["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# Show the agent
try:
    from IPython.display import Image, display # type: ignore
    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
except ImportError:
    print("IPython not installed, skipping graph display.")

messages: list[AnyMessage] = [HumanMessage(content="Get financial data for the following tickers")]
message_state: MessageState = {"messages": messages, "llm_calls": 0}
messages_output = agent.invoke(message_state)
for m in messages_output["messages"]:
    m.pretty_print()