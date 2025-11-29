from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.tool_context import ToolContext


from .utility import (
    save_state_after_agent_callback, 
    update_agent_state_for_clarification,
    update_agent_state_for_profile,
    fetch_page_details, 
    update_agent_state,
    planner_before_agent_callback,
    writer_before_agent_callback,
    prepare_verify_pairs,
    apply_verification_updates,
    save_user_profile,
    save_search_results,
    create_uuid_for_search_results,
    writer_before_agent_callback, 
    writer_after_agent_callback
)
from .schema import SummaryOutput, NewsletterSections, clarifications_needed, NewsletterOutput, NewsletterProfileOutput, VerificationOutput
from .prompt import NEWSLETTER_PROMPT

MODEL = "gemini-2.5-flash"

requirement_agent = LlmAgent(
    model=MODEL,
    name="Newsletter_Request_Clarification",
    instruction="""
        You are the request clarification Agent for a weekly AI/GenAI Newsletter. 
        Your responsibility is to interact with the user to collect sufficient detail request for the newsletter:
        1. If any ambiguity in the request, please ask questions to understand the detail info about the requests (short vs long for the length; 
            prioritize updates from specific sources (e.g., OpenAI, Google DeepMind, Meta, Anthropic, or academic conferences like NeurIPS); more on technical or 
            business implications; citations for each sentence or each item; professional or informal style ). 
        2. When asking the clarifying questions, please add the number sequence in front of each question and formalize those questions for user easy to read, understand and respond.  
        3. Include the user's answers to generate more personalized and detailed request prompt. Please summarize user's answers to the clarification questions 
            and summarize the updated request and requirements to user. 
        4. The output should be detailed request that addresses all the ambiguity to the planner agent, set it to detailed_request field. If you collect all the   
            answers to the clarification questions, please set request_clarification_done to be true; else, please wait for user's input to the clarification questions.
        """,
    output_key = 'request_clarification',
    output_schema=clarifications_needed,
    # before_agent_callback=before_agent_callback_clarification,
    after_agent_callback=update_agent_state_for_clarification
)


prompt_profile = """You are a user profiling agent for an AI/ML newsletter.

The LAST user message contains:
- user self-description
- user's email
- user's detail request
- Optionally, a list of preferred or trusted sources (e.g. 'openai.com, arxiv.org, anthropic.com').

Task:
1. Read the message.
2. Infer:
   - email (string)
   - technical_level in ["beginner", "intermediate", "expert"]
   - preferred_sources: a list of domain strings like ["openai.com", "arxiv.org"].
     If the user did not provide any, use an empty list [].
   - a detail request elaborating user's requirement for the newsletter

3. Call the save_user_profile tool with this JSON to persist it.

4. Return ONLY valid JSON (no prose). Example:

{
  "email": "user@example.com",
  "technical_level": "expert",
  "preferred_sources": ["openai.com", "arxiv.org"]
  "detailed_request": 
}
"""

profile_agent = LlmAgent(
    model=MODEL,
    name="personal_profile_agent",
    instruction=prompt_profile,
    output_key = 'profile',
    output_schema=NewsletterProfileOutput,
    after_agent_callback=update_agent_state_for_profile,
    )

planner_agent = LlmAgent(
    model=MODEL,
    name="Newsletter_Planner",
    instruction=NEWSLETTER_PROMPT,
    output_schema= NewsletterSections,
    after_agent_callback=update_agent_state,
    output_key = 'plan',
    # before_agent_callback=planner_before_agent_callback,
    )



executive_search_agent = LlmAgent(
    name="executive_search_agent",
    model=MODEL,
    instruction="""
        You have the below request from user: 
        {detailed_request}

        You are a search assistant, the first step of the executive summary agent.

        GOAL:
        For each topic below, find at most **1** highly relevant result
        from the **last 7 days only**.

        HOW TO USE google_search:
        - When you call the `google_search` tool, always:
          - Use the topic string as the query.
          - Constrain results to the last 7 days (for example by using a
            recency / date filter such as `recency_days: 7` if the tool supports it,
            or by adding suitable time filters to the query like "past 7 days").
        - After you get search results, infer `publish_date` from the page
          (snippet or content), and **discard** any result older than 7 days.

        For each topic, return a JSON array where each item includes:
            - topic: the topic string (include the time window, e.g. "LLM safety updates (last 7 days)")
            - title: the title or snippet of the result  
            - url: the URL of the result  
            - publish_date: the publish date inferred from the page (ISO if possible)
            - uuid: unique identification for each result
            - short_summary: a short summary of the web page

        Here are the topics to search:
        {search_queries}

        OUTPUT RULES:
        - If no suitable result was found in the last 7 days for a topic,
          omit that topic or set url to null and short_summary to "".
        - Make sure your output is **valid JSON only** (no extra commentary),
          and **only keep results within the last 7 days**.
        """,
    tools=[google_search],
    output_key="search_results_executive",
    after_model_callback=create_uuid_for_search_results,
    after_agent_callback=save_search_results,
)

executive_fetch_agent = LlmAgent(
    name="executive_fetch_agent",
    model=MODEL,  # or your Gemini-2 model
    instruction="""
        You are a wetsite url fetch assistant. For each webpage below, perform a content fecth (using the fetch_page_details tool). 
        Then, return a JSON array where each item includes (can just return the results from tool call):
            - topic
            - google_title
            - url
            - uuid: unique identification for each result, same to the uuid from search agent for reference
            - final_url
            - canonical_url
            - page_title (from the page `<title>` tag)
            - summary: first paragraph (or first few sentences)
            - full_text: the entire text of the webpage

        Here are the webpages to fetch:
        {search_results_executive}
        Make sure your output is **valid JSON only** (no extra commentary).
        """,
    tools=[fetch_page_details],
    output_key="fetch_results_executive",
    after_agent_callback=save_state_after_agent_callback,
)


executive_summary_agent = LlmAgent(
    name="executive_summary_agent",
    model=MODEL,  # or whatever LLM you're using
    instruction="""
        {executive_summary_agent_prompt}
        You are an AI research summarizer. You will get  a JSON list of objects below, 
        each containing a 'topic', 'title', 'final_url', 'uuid', and 'full_text' (the entire text of a webpage). 
        Your job is to o produce a clear, concise summary for each page’s full_text.
        Input list : {fetch_results_executive}
        After calling the tool, return a JSON list of dictionaries, each with:
        - topic (string)
        - title (string)
        - final_url (string)
        - uuid: unique identification for each result, same to the uuid from fetch agent for reference
        - publish_date
        - summary (string) — summarizing the full_text in your own words
        - also set 'summary_done' to be True

        Respond with **only valid JSON**, no extra text.
        """,
    # before_model_callback= get_valid_full_text
    # tools=[summarize_tool],
    output_key="executive_summary",
    output_schema=SummaryOutput,
    after_agent_callback=save_state_after_agent_callback
)





# newsletter_agent.py
# import json
# import time
# from typing import List, Optional
# from pydantic import BaseModel, Field
# from google.adk.agents import LlmAgent
# from google.adk.agents.callback_context import CallbackContext
# from google.adk.models import LlmResponse
# from google.adk.sessions import InMemorySessionService
# from google.adk.runners import Runner
# from google.genai import types

# MODEL = "gemini-2.5-flash"  # adapt if you want a different model

def exit_loop(tool_context: ToolContext):
    """
    Signal to the LoopAgent that verification is complete and
    we can stop iterating. This is used as a tool by Newsletter_Verifier.
    """
    tool_context.actions.escalate = True
    return {"status": "loop_ended"}

# -------------------------
# The Agent: instruction & wiring
# -------------------------
INSTRUCTION = """
You are an expert newsletter writer for an AI/GenAI weekly briefing aimed at senior product and business readers in healthcare insurance.
The detailed request from the user is : {profile}
INPUT: The agent will be provided two structured summaries:
 - {executive_summary}  (list of dict with keys: topic, title, url, uuid, publish_date, summary)

TASK:
Using those inputs, produce JSON only that matches the NewsletterOutput schema exactly:
- newsletter_title (short headline)
- date (YYYY-MM-DD)
- short_blurb (1 sentence)
- executive_summary (list of items; each item: heading, body, url, uuid; draw from executive_summary;  
    please refer to {section_outline} to divide each section to     
    subsection if the total items are more than 3. Try to keep the mamximum number of items for each subsection to 3  be at most)
- business_implications (list of items; each item: heading, body, url, uuid; emphasize implications for healthcare insurance; please refer to {section_outline} to divide each section to subsection if the total items are more than 3. Try to keep the mamximum number of items for each subsection to
    be at most 3. )
- citations (list of source URLs, got from url from executive_sumamry and business_summary; deduplicate)
- tl_dr (3 concise bullet lines separated by '\\n')
- call_to_action (single paragraph advising a practical next step for product leaders)

REQUIREMENTS:
1. Output valid JSON ONLY, nothing else. 
2. The date should be today in YYYY-MM-DD format (use session state if provided: callback_context.state['newsletter_date'] else infer from user's request).
3. Keep each heading <= 15 words; Each item body <= 200 words.
4. For the three sections of technical_highlitghts and business_implications, please refer to {section_outline}
    to divide each section to subsection if the total items are more than 3. Try to keep the mamximum number of items for each subsection to
    be at most 3. 
5. For citations, use the provided url fields; if AnyUrl appears, output its string form.
6. When synthesizing, prefer facts and avoid hallucination; if a fact is only in one summary, mark it as "reported by source".
"""

# NewsletterWriter = LlmAgent(
#     name="NewsletterWriter",
#     model=MODEL,
#     instruction=INSTRUCTION,
#     output_key="newsletter_result",
#     output_schema=NewsletterOutput,
#     # disallow_transfer_to_parent=True,
#     # disallow_transfer_to_peers=True,
#     before_agent_callback=writer_before_agent_callback,
#     after_agent_callback=save_state_after_agent_callback # prepare_verify_pairs
# )
from .utility import writer_before_agent_callback, writer_after_agent_callback

NewsletterWriter = LlmAgent(
    name="NewsletterWriter",
    model=MODEL,
    instruction=INSTRUCTION,
    output_key="newsletter_result",
    output_schema=NewsletterOutput,
    # disallow_transfer_to_parent=True,
    # disallow_transfer_to_peers=True,
    before_agent_callback=writer_before_agent_callback,
    after_agent_callback=writer_after_agent_callback,
)


verification_agent = LlmAgent(
    name="Newsletter_Verifier",
    model=MODEL,
    instruction="""
You are a precise verification assistant. You will be given a list of sentences and references from the source documents.

For each sentence:
- Output an object with:
  - sentence (string)
  - uuid (string)
  - accuracy_or_not (true/false)
  - modified_version (string)
  - justification (string)

Rules for `modified_version`:
- If accuracy_or_not is true and no change is needed, set modified_version to be EXACTLY the original sentence.
- If accuracy_or_not is false, set modified_version to a conservative corrected version.
- Never use null for modified_version; always return a string.

Input:
{verification_pairs}

OUTPUT FORMAT:
JSON array ONLY, e.g.:
[
  {
    "sentence": "...",
    "uuid": "...",
    "accuracy_or_not": true,
    "modified_version": "...",  // original or corrected sentence
    "justification": "..."
  }
]
    """,
    output_key="verification_result",
    before_agent_callback=prepare_verify_pairs,
    after_agent_callback=apply_verification_updates,
    output_schema=VerificationOutput,
    tools=[exit_loop],
)
