from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from .states import WorkflowState


model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

def judge_analyst(state: WorkflowState) -> dict:
    """
    Judge synthesizes bull and bear analysts' theses and generates a complete report
    """
    print(f"[Judge] Synthesizing theses...")
    print(f"\n[Judge] Bull thesis: {state.get('bull_thesis', '')[:80]}...")
    print(f"\n[Judge] Bear thesis: {state.get('bear_thesis', '')[:80]}...")
    systemMessage = f"Act as an expert stock research analyst grounded in value investing principles. You are presented with bull and bear theses for {state["company"]} ({state["ticker"]}) and you are tasked with synthesizing both sides' research to create an investment thesis for {state["company"]}. This research should take an opinionated view based on the evidence presented by the bull and bear analysts and should include the hallmarks of value investing like discounted cash flow based on probability of outcomes and investability based on margin of safety. Do NOT call any tools. Respond with plain text only."

    response = model.invoke([SystemMessage(systemMessage)])
    return {"judge_decision": response.content}

