from dotenv import load_dotenv
from utils.states import GenerateAnalystsState
from typing import Literal
from langgraph.graph import END



load_dotenv()

# conditional edge

def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", END]:
    """Return the next node to execute"""

    human_analyst_feedback = state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "create_analysts"
    return END