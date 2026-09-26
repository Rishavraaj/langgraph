from dotenv import load_dotenv
from utils.states import GenerateAnalystsState
from utils.models import llm
from utils.objects import Analyst, Perspective
from utils.prompts import analyst_instructions
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

load_dotenv()

def create_analysts(state: GenerateAnalystsState):
    """Create analyst"""

    topic = state["topic"] 
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    #Enforce structured output
    structured_llm = llm.with_structured_output(Perspective)
    
    #System_message
    system_message = analyst_instructions.format(topic=topic, human_analyst_feedback=human_analyst_feedback, max_analysts=max_analysts)

    #Generate analysts
    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Please generate the set of analysts")])

    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """This is where humans give the feedback about the analysts given"""

    feedback = interrupt({
        "question": "Are these analyst okay for you?",
        "analyst": [
          analyst.model_dump() 
            if hasattr(analyst, "model_dump")
            else analyst 
            for analyst in state.get("analysts", [])
        ],
        "instructions": "Return feedback to regenerate analysts or return empty/perfect/continue/okay to approve an continue the graph"
    })

    if feedback is None: 
        return {"human_analyst_feedback": None}
    
    if isinstance(feedback, str):
        feedback = feedback.strip()

        if feedback == "":
           return {"human_analyst_feedback": None}

        if feedback.lower() in {"perfect", "okay", "continue", "yes"}:
           return {"human_analyst_feedback": None}

        return {"human_analyst_feedback": feedback}

    return {"human_analyst_feedback": None}