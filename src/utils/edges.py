from dotenv import load_dotenv
from utils.states import GenerateAnalystsState, InterviewState, ResearchGraphState
from typing import Literal
from langgraph.graph import END
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langchain.messages import HumanMessage



load_dotenv()

# conditional edge

def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", END]:
    """Return the next node to execute"""

    human_analyst_feedback = state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "create_analysts"
    return END

def routes_messages(state: InterviewState, name: str="expert"):
    """Route between question ans answer"""

    #get messages
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    #check the number if expert answer
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])

    if num_responses >= max_num_turns:
        return "save_interview"
    
    return "ask_question"

def initiate_all_interviews(state: ResearchGraphState):
    """This is the map state where we run each interview in sub graph using send API"""

    human_analyst_feedback = state.get("human_analyst_feedback")

    if human_analyst_feedback:
        return "create_analyst"
    else: 
        topic = state["topic"]
        return[
            Send("conduct_interview", {
                "analyst": analyst,
                "messages": [HumanMessage(content=f"so you said you were writing an article on {topic}")]
            }) for analyst in state["analysts"]
        ]