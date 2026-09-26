analyst_instructions= """
Your are tasked with creating a set of AI analyst personas.Follow this instructions 
care fully.
1. First, review the research topic:
{topic}
2. Examine any editorial feedback that has been optionally provided to guide creation of the analyst
{human_analyst_feedback}
3. Determine the most interesting theme based upon documents and / or feedback above.
4. Pick the top {max_analysts} themes.
5. Assign one analyst to each theme.
"""