from dotenv import load_dotenv
from utils.states import GenerateAnalystsState, InterviewState, ResearchGraphState
from utils.models import llm
from utils.objects import Analyst, Perspective, SearchQuery
from utils.prompts import analyst_instructions, question_instruction, search_instructions, answer_instructions, section_writer_instructions, intro_conclusion_instructions, report_writer_instructions
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch
from langchain_core.messages import get_buffer_string

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

def generate_questions(state: InterviewState):
    """node to generate the questions"""

    # get state analyst
    analyst = state["analyst"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)
    
    messages = state["messages"]

    # generate question
    system_message = question_instruction.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    return {"messages": [question]}

def search_web(state: InterviewState):
    """Retrieve docs from the web"""

    #Search query
    structured_llm = llm.with_structured_output(SearchQuery)

    #Search Instruction
    search_instruction_system_message = SystemMessage(content=search_instructions)
    tavily_search = TavilySearch(max_results=3)
    search_query = structured_llm.invoke([search_instruction_system_message]+state["messages"]) 

    data = tavily_search.invoke({"query": search_query.search_query})

    search_docs = data.get("results", data)

    #format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs 
        ]
    )

    return {"context" : [formatted_search_docs]}

def search_web2(state: InterviewState):
    """Retrieve docs from the web"""

    #Search query
    structured_llm = llm.with_structured_output(SearchQuery)

    #Search Instruction
    search_instruction_system_message = SystemMessage(content=search_instructions)
    tavily_search = TavilySearch(max_results=3)
    search_query = structured_llm.invoke([search_instruction_system_message]+state["messages"]) 

    data = tavily_search.invoke({"query": search_query.search_query})

    search_docs = data.get("results", data)

    #format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs 
        ]
    )

    return {"context" : [formatted_search_docs]}

def generate_answer(state: InterviewState):
    """Node to answer a question"""

    #get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    if isinstance(analyst,dict):
        analyst = Analyst.model_validate(analyst)

    #answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)

    #name the message coming from the expert
    answer.name = "expert"

    #append to the state
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """save interviews"""

    messages = state["messages"]

    interview = get_buffer_string(messages)

    return {"interview" : interview}

def write_section(state: InterviewState):
    """Node to answer a question"""

    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)
    
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section : {context}")])

    return {"sections": [section.content]}

def write_report(state: ResearchGraphState):
    """Write the final report"""

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Write a report based upon these memos.")])

    return {"content": report.content}

def write_introduction(state: ResearchGraphState):

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Write a report introduction")])

    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):

    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    system_message = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Write a report conclusion")])

    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}
