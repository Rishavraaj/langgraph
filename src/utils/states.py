from typing_extensions import TypedDict, NotRequired
from typing import Optional, List
from utils.objects import Analyst

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic.
    max_analysts: int # Number of analysts.
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback for what is generated.
    analysts: NotRequired[List[Analyst]] # Lists of all our analysts.
