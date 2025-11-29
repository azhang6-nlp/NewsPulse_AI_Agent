from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.tool_context import ToolContext

from .utility import (save_state_after_agent_callback, 
                      update_agent_state_for_clarification,
                      update_agent_state_for_profile,
                      fetch_page_details, 
                      update_agent_state_planner,
                      writer_before_agent_callback,
                      prepare_verify_pairs,
                      apply_verification_updates,
                      save_user_profile,
                      save_search_results,
                      create_uuid_for_search_results,
                      load_user_profile,         
                      semantic_search_articles,  
                      parse_feedback,              
                      send_newsletter_email,
                      update_agent_state_for_recommender,
                      writer_before_agent_callback, 
                      writer_after_agent_callback
                      )
from .schema import SummaryOutput, NewsletterSections, clarifications_needed, NewsletterOutput, NewsletterProfileOutput, VerificationOutput
from .prompt import PLANNER_PROMPT, WRITER_INSTRUCTION

MODEL = "gemini-2.5-flash"

# --- Shared State Keys ---
STATE_USER_PROFILE = "profile"
STATE_REFINED_TOPICS = "refined_topics"
STATE_SUMMARIES = "executive_summary" # Using your existing key for summaries

# -----------------------------------------------------------
# 1. Clarification Agent (Unchanged)
# -----------------------------------------------------------

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
        4. The output should be detailed request that addresses all the ambiguity to the planner agent, set it to {detailed_request?} field. If you collect all the   
            answers to the clarification questions, please set request_clarification_done to be true; else, please wait for user's input to the clarification questions.
        """,
    output_key = 'request_clarification',
    output_schema=clarifications_needed,
    after_agent_callback=update_agent_state_for_clarification
)


# -----------------------------------------------------------
# 2. Profile Agent (Updated to check for existing profile)
# -----------------------------------------------------------

prompt_profile = """You are a user profiling agent for an AI/ML newsletter.

The LAST user message contains:
- user self-description
- user's email
- user's detail request
- Optionally, a list of preferred or trusted sources (e.g. 'openai.com, arxiv.org, anthropic.com').

Task:
1. Attempt to load an **existing profile** using the email via the load_user_profile tool.
2. If a profile exists, use it as the base and only update fields explicitly mentioned in the LAST user message (e.g., new request details or sources).
3. If no profile exists, infer all fields (email, technical_level, preferred_sources, detailed_request) from the message.
4. Call the save_user_profile tool with the final, merged JSON to persist it for future runs.

5. Return ONLY valid JSON (no prose). Example:

{
  "email": "user@example.com",
  "technical_level": "expert",
  "preferred_sources": ["openai.com", "arxiv.org"],
  "detailed_request": "..."
}
"""


profile_agent = LlmAgent(
    model=MODEL,
    name="personal_profile_agent",
    instruction=prompt_profile,
    tools=[save_user_profile, load_user_profile], 
    output_key = STATE_USER_PROFILE,
    # 💥 DELETE THIS LINE: output_schema=NewsletterProfileOutput, 
    after_agent_callback=update_agent_state_for_profile,
)

# -----------------------------------------------------------
# 3. Historical Recommender Agent (NEW)
# -----------------------------------------------------------

historical_recommender_agent = LlmAgent(
    name="HistoricalRecommenderAgent",
    model=MODEL,
    description="Adjusts the user's topics to include novelty and drift based on related articles in the vector database.",
    tools=[semantic_search_articles],
    instruction=f"""
You are the Historical Recommender. Your goal is to slightly modify the user's
topic list (from profile['detailed_request']) to prevent stagnation and introduce related, novel concepts for today's search.

Inputs from state:
- The user profile in state["{STATE_USER_PROFILE}"]

Task:
1. Analyze the user's current interests and detailed request from the profile.
2. Suggest 1-2 new, related topics that would introduce novelty but remain relevant.
3. Combine these 1-2 new topics with the existing request/topics.
4. Output ONLY the refined detailed request as a JSON object, keeping the original structure but with enriched topics/queries for the Planner Agent.
5. Do NOT call the semantic_search_articles tool; only use its description to guide your suggestions.

Return ONLY a JSON object that matches the structure of the detailed request:
{{
  "detailed_request_updated": "Original request text, now enhanced with 1-2 related, novel concepts, e.g., '... (and include Model Drift as a topic).'"
}}
""",
    output_key=STATE_REFINED_TOPICS,
    after_agent_callback=update_agent_state_for_recommender,
)


# -----------------------------------------------------------
# 4. Planner Agent 
# -----------------------------------------------------------

planner_agent = LlmAgent(
    model=MODEL,
    name="Newsletter_Planner",
    instruction=PLANNER_PROMPT,
    output_schema= NewsletterSections,
    after_agent_callback=update_agent_state_planner,
    output_key = 'plan',
    )


# -----------------------------------------------------------
# 5. Executive Search Agent 
# -----------------------------------------------------------

executive_search_agent = LlmAgent(
    name="executive_search_agent",
    model=MODEL,
    instruction=""" You have the below request from user: 
        {detailed_request}

        You are a search assistant, the first step of the executive summary agent.

        GOAL:
        For each topic below, perform a Google Search (using the google_search tool) to find up to **1** highly relevant result from the date range specified in the request**.

        HOW TO USE google_search:
        - When you call the `google_search` tool, always:
          - Use the topic string as the query.
          - Constrain results to the date range specified in the request (for example by using a
            recency / date filter such as `recency_days: 14` if the tool supports it,
            or by adding suitable time filters to the query like "past 14 days" if the user is interested in the last two weeks' update).
        - After you get search results, infer `publish_date` from the page
          (snippet or content), and **discard** any result beyond the date range of request.

        For each topic, return a JSON array where each item includes:
            - topic: the topic string (include the time window, e.g. "LLM safety updates (last 14 days)")
            - title: the title or snippet of the result  
            - url: the URL of the result  
            - publish_date: the publish date inferred from the page (ISO if possible)
            - uuid: unique identification for each result
            - short_summary: a short summary of the web page

        Here are the topics to search:
        {search_queries}

        OUTPUT RULES:
        - Make sure your output is **valid JSON only** (no extra commentary),
          and **only keep results within the last several days according to the request**.
        """,
    tools=[google_search],
    output_key="search_results_executive",
    after_model_callback=create_uuid_for_search_results,
    after_agent_callback=save_search_results,
)

# -----------------------------------------------------------
# 6. Executive Fetch Agent 
# -----------------------------------------------------------

executive_fetch_agent = LlmAgent(
    name="executive_fetch_agent",
    model=MODEL,
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


# -----------------------------------------------------------
# 7. Executive Summary Agent 
# -----------------------------------------------------------

executive_summary_agent = LlmAgent(
    name="executive_summary_agent",
    model=MODEL,
    instruction="""
        {executive_summary_agent_prompt?}
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
    output_key="executive_summary",
    output_schema=SummaryOutput,
    after_agent_callback=save_state_after_agent_callback
)

def exit_loop(tool_context: ToolContext):
    """
    Signal to the LoopAgent that verification is complete and
    we can stop iterating. This is used as a tool by Newsletter_Verifier.
    """
    tool_context.actions.escalate = True
    return {"status": "loop_ended"}


# -----------------------------------------------------------
# 8. Newsletter Writer 
# -----------------------------------------------------------

NewsletterWriter = LlmAgent(
    name="NewsletterWriter",
    model=MODEL,
    instruction=WRITER_INSTRUCTION,
    output_key="newsletter_result",
    output_schema=NewsletterOutput,
    before_agent_callback=writer_before_agent_callback,
    after_agent_callback=writer_after_agent_callback,
)

# -----------------------------------------------------------
# 9. Verification Agent 
# -----------------------------------------------------------

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


# -----------------------------------------------------------
# 10. newsletter_dispatcher Agent (NEW)
# -----------------------------------------------------------


newsletter_dispatcher = LlmAgent(
    model="gemini-2.5-flash",
    name="newsletter_dispatcher",
    description=(
        "Converts newsletter json to HTML and sends it to an email using "
        "the tool of send_newsletter_email."
    ),
    instruction=(
        "You MUST:\n"
        "1. Call send_newsletter_email exactly once. get the to_email from {email} \n"
        "3. Pass to_email, subject, and the newsletter json in {newsletter_updated}.\n"
        "Return ONLY the tool call result.\n"
    ),
    tools=[send_newsletter_email],
)
# -----------------------------------------------------------
# 11. Feedback Agent (NEW)
# -----------------------------------------------------------

feedback_agent = LlmAgent(
    name="FeedbackAgent",
    model=MODEL,
    description="Interprets user feedback and updates preferences in the profile for future runs.",
    tools=[parse_feedback, save_user_profile], # Must include both parsing and saving tools
    instruction=f"""
You are the FeedbackAgent for the AI newsletter.

The LAST user message contains free-text feedback about the newsletter.
Also available in state is the current profile: state["{STATE_USER_PROFILE}"].

Steps:
1. Call parse_feedback(raw_email_text) with the *entire* last user message.
2. Combine the parsed feedback with the existing profile to update fields like technical_level, preferred_sources, or the detailed_request.
   *Example: If feedback says "too technical," reduce the technical_level.*
3. Call save_user_profile(updated_profile) to persist these changes for the next run.
4. Return ONLY the updated profile as JSON.

If no meaningful feedback is present, just return the unchanged profile.
""",
    output_key=STATE_USER_PROFILE,
)