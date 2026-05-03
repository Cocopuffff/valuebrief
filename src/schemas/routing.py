"""
routing.py
~~~~~~~~~~
LangGraph node name enum.
"""

from enum import Enum


class AgentNode(str, Enum):
    SUPERVISOR = "supervisor"
    BEAR = "bear_analyst"
    BULL = "bull_analyst"
    JUDGE = "judge_analyst"
    REPORT_GENERATOR = "report_generator"
    CURATOR = "curator_agent"

    RESEARCH_TOOL_NODE = "research_tool_node"
