from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Optional, List
from utils.objects import Analyst
from langgraph.graph import MessagesState
import operator

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic.
    max_analysts: int # Number of analysts.
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback for what is generated.
    analysts: NotRequired[List[Analyst]] # Lists of all our analysts.


class InterviewState(MessagesState):
    max_num_turns: int # Number turns of questions
    context: Annotated[list, operator.add] # source of docs
    analyst: Analyst # my analyst
    interview: str # interview transcript
    sections: list # final key we duplicated in outer state fro send() api