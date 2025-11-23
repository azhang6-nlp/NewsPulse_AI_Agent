import requests
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

import requests
from bs4 import BeautifulSoup, Comment
import json
import re
from typing import List

# Common article container tags/classes/ids
CONTENT_HINTS = [
    "article", "main", "content", "post", "entry", "body-content",
    "post-content", "article-body", "main-content", "StoryBodyCompanion",
    "Section1", "RichText", "td-post-content"
]

def update_agent_state_for_clarification(callback_context):
    if callback_context.state['request_clarification']:
        clarifications_needed = callback_context.state['request_clarification'].get('clarifications_needed',[])
        questions = [item['question'] for item in clarifications_needed]
        answers = [item['answer'] for item in clarifications_needed if len(item['answer'])>1 ]
        if len(questions) > 0:
            if len(answers) >0:
                n = min(len(questions), len(answers))
                callback_context.state['formatted_questions'] = 'Based on your input, the updated request is: ' + callback_context.state['request_clarification'].get('detailed_request','')
            else:
                callback_context.state['formatted_questions'] = [f'Question {q} \n ' for q in questions]
    save_state_after_agent_callback(callback_context)


def clean_text(txt):
    # Collapse whitespace, remove artifacts
    return re.sub(r"\s+", " ", txt).strip()

def extract_main_content(soup, page_title=None):
    # 1. Remove scripts, styles, footers, headers, ads, nav, etc.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 2. Try structural content extraction (containers that likely hold article text)
    candidates = []

    body = soup.body or soup
    for tag in body.find_all():
        id_class = " ".join((tag.get("id") or "").split() + (tag.get("class") or []))
        if any(hint.lower() in id_class.lower() for hint in CONTENT_HINTS):
            txt = clean_text(tag.get_text(" ", strip=True))
            if len(txt) > 300:
                candidates.append((len(txt), txt))

    if candidates:
        candidates.sort(reverse=True)  # largest content first
        return candidates[0][1]  # best match

    # 3. Fallback: find the most text-dense block
    blocks = []
    for tag in body.find_all(["div", "section", "article", "p"]):
        txt = clean_text(tag.get_text(" ", strip=True))
        if len(txt) > 200:  # discard small irrelevant blocks
            blocks.append((len(txt), txt))

    if blocks:
        blocks.sort(reverse=True)
        best = blocks[0][1]
        # Optional: keep only paragraphs relevant to title keywords
        if page_title:
            keywords = [w.lower() for w in page_title.split() if len(w) > 4]
            filtered = "\n".join(
                p for p in best.split(". ")
                if any(k in p.lower() for k in keywords)
            )
            if len(filtered) > 0.3 * len(best):
                return filtered

        return best

    # 4. Last fallback: entire body text
    return clean_text(body.get_text(" ", strip=True))


def fetch_page_details(pages: list[dict]) -> dict:
    """
    Fetch detailed information for a list of web pages.

    Given a list of page descriptors (each containing keys like
    'topic', 'title', and 'url', 'uuid'), this function issues HTTP GET
    requests to retrieve each page, then parses its HTML content to
    extract useful metadata.

    Returns a dictionary with the value having a list of dictionaries, each containing:

      - topic: the topic/category associated with the page  
      - google_title: the title as returned by Google Search  
      - url: the URL of the page  
      - uuid: unique identification passed from the input
      - page_title: the content of the HTML <title> tag (if available)  
      - summary: the first several paragraph text, trimmed of whitespace  
      - full_text: the entire visible textual content of the page  

    In case of an error fetching or parsing a page, the dictionary will
    include an 'error' field with an error message.

    Args:
        pages (list[dict]): A list of dictionaries, each describing a page.
            Expected keys:
              - 'topic' (str): The topic associated with this page.
              - 'title' (str): The title from Google search.
              - 'url' (str): The URL of the page.
              - 'uuid' (str): The unique identification of the page

    Returns:
        Dict[str, str in json format].

    Raises:
        None: All network or parsing errors are caught and encoded in the
        result list as error messages.
    """
    
    results = []

    for page in pages:
        topic = page.get('topic')
        title = page.get('title')
        url = page.get('url')
        uuid = page.get('uuid')

        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            final_url = resp.url
        except:
            # unreachable
            results.append({
                "topic": topic,
                "google_title": title,
                "url": url,
                "uuid": uuid,
                "error": "Failed initial GET"
            })
            continue

        try:
            resp = requests.get(final_url, timeout=6)
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            # Canonical URL
            canonical_tag = soup.find("link", rel="canonical")
            canonical_url = canonical_tag["href"] if canonical_tag and canonical_tag.has_attr("href") else None

            # Page <title>
            page_title = soup.title.string.strip() if soup.title else None

            # 🔥 Extract main content (removes boilerplate)
            main_text = extract_main_content(soup, page_title=page_title)

            # Summary (first ~1000 chars)
            summary = main_text[:1000]

            results.append({
                "topic": topic,
                "google_title": title,
                "url": url,
                "uuid": uuid,
                "final_url": final_url,
                "canonical_url": canonical_url,
                "page_title": page_title,
                "summary": summary,
                "full_text": main_text,
            })

        except Exception as e:
            print(f"Error fetching content for {final_url}: {e}")
            results.append({
                "topic": topic,
                "google_title": title,
                "url": url,
                "uuid": uuid,
                "final_url": final_url,
                "canonical_url": '',
                "page_title": '',
                "summary": '',
                "full_text": '',
                "error": str(e),
            })

    return results


def update_agent_state(callback_context: CallbackContext):
    callback_context.state.get('search_queries_for_executive',[])
    callback_context.state.get('search_queries_for_Industry_Implications',[])
    callback_context.state.get('executive_summary_agent','')
    callback_context.state.get('industry_implications_agent','')
    callback_context.state.get('section_outline',{})
    print(callback_context.state['plan'])
    if callback_context.state['plan']:
        # callback_context.state['plan'] = json.loads( callback_context.state['plan'].split('```json')[-1].replace('```','') )
        callback_context.state["search_queries_for_executive"] = callback_context.state['plan'].get('search_queries_for_executive',[])
        callback_context.state["search_queries_for_Industry_Implications"] = callback_context.state['plan'].get('search_queries_for_Industry_Implications',[])
        prompts = callback_context.state['plan'].get('task_delegation_plan',{})
        callback_context.state['executive_summary_agent_prompt'] = prompts.get('search_results_executive','')
        callback_context.state['industry_implications_agent_prompt'] = prompts.get('industry_implications_agent','')
        callback_context.state['section_outline'] = callback_context.state['plan'].get('section_outline',{})
    # state_dict = callback_context.state.to_dict()
    # print(f"[after_model] Current session state: {state_dict.keys()}")

    save_state_after_agent_callback(callback_context)

def planner_before_agent_callback(callback_context):
    """
    This runs before planner_agent executes. If requirement step is not done,
    raise a controlled exception to prevent planner from running, or return
    a short LlmResponse directing the agent runner not to proceed.
    """
    # Check pipeline flags in state
    state_dict = callback_context.state.to_dict()
    print(f"[planner_before_agent_callback] Current session state: {state_dict.keys()}")
    clarifications = callback_context.state.get("request_clarification", {})
    request_clarification_done = clarifications.get('request_clarification_done', False)


    if not clarifications or not request_clarification_done:
        # Prevent planner from running. Two common patterns:
        # 1) raise an exception the runner will catch and stop the pipeline.
        # 2) set a flag and return a sentinel response.

        # Option A: raise to stop progression (make sure your runner catches it and stops)
        raise RuntimeError("Pipeline paused: requirement_agent needs clarifications")

        # Option B: alternatively, set a state field and return a dummy LlmResponse
        # (less standard; depends on ADK internals)
    # else: planner proceeds
    else:
        callback_context.state['detailed_request'] = clarifications.get('detailed_request', '')



# -------------------------
# After-callback to save markdown + store JSON in state
# -------------------------
def after_agent_callback_save_md(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:
    """
    - Parse model output (assumes model returns JSON matching NewsletterOutput)
    - Save JSON to callback_context.state["newsletter_json"]
    - Also write a human-friendly Markdown file newsletter_<session>_<ts>.md
    - Overwrite llm_response text to the JSON string (so ADK UIs show structured output)
    """
    text = ""
    try:
        if llm_response.content and llm_response.content.parts:
            text = llm_response.content.parts[0].text
    except Exception:
        text = ""

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        # try to read from state (some ADK setups already put structured output in state)
        parsed = callback_context.state.get("newsletter_json_raw")

    if isinstance(parsed, dict):
        # validate/normalize using the Pydantic model
        try:
            out = NewsletterOutput.model_validate(parsed)  # pydantic v2
        except Exception:
            try:
                out = NewsletterOutput(**parsed)  # fallback for v1-style
            except Exception as e:
                # If validation fails, store raw and return
                callback_context.state["newsletter_json_raw"] = parsed
                print("Newsletter parse/validation failed:", e)
                return llm_response

        # Save JSON into session state
        callback_context.state["newsletter_json"] = out.model_dump() if hasattr(out, "model_dump") else out.dict()

        # Build Markdown
        md_lines = []
        md_lines.append(f"# {out.newsletter_title}\n")
        md_lines.append(f"*Date: {out.date}*\n")
        md_lines.append(f"**{out.short_blurb}**\n")

        md_lines.append("## Executive Summary\n")
        for item in out.executive_summary:
            md_lines.append(f"### {item.heading}\n")
            md_lines.append(item.body + "\n")

        md_lines.append("## Technical Highlights\n")
        for item in out.technical_highlights:
            md_lines.append(f"### {item.heading}\n")
            md_lines.append(item.body + "\n")

        md_lines.append("## Business & Industry Insights\n")
        for item in out.business_implications:
            md_lines.append(f"### {item.heading}\n")
            md_lines.append(item.body + "\n")

        md_lines.append("## TL;DR\n")
        md_lines.append(out.tl_dr + "\n")

        md_lines.append("## Citations / Sources\n")
        for c in out.citations:
            md_lines.append(f"- {c}")

        md_lines.append("\n## Call to Action\n")
        md_lines.append(out.call_to_action + "\n")

        md_text = "\n".join(md_lines)

        # Write to file named by session id and timestamp
        try:
            session_id = callback_context._invocation_context.session.id
        except Exception:
            session_id = "local"
        ts = int(time.time())
        filename = f"newsletter_{session_id}_{ts}.md"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_text)
            callback_context.state["newsletter_md_path"] = filename
        except Exception as e:
            callback_context.state["newsletter_md_error"] = str(e)

        # Overwrite the response text with normalized JSON for cleaner UI display
        try:
            llm_response.content.parts[0].text = json.dumps(callback_context.state["newsletter_json"], ensure_ascii=False, indent=2)
        except Exception:
            pass

    else:
        # store the raw text for debugging
        callback_context.state["newsletter_raw_text"] = text
    
    # save_state_after_agent_callback(callback_context)
    return llm_response

def writer_before_agent_callback(callback_context):
    """
    This runs before planner_agent executes. If requirement step is not done,
    raise a controlled exception to prevent planner from running, or return
    a short LlmResponse directing the agent runner not to proceed.
    """
    # Check pipeline flags in state
    state_dict = callback_context.state.to_dict()
    print(f"[planner_before_agent_callback] Current session state: {state_dict.keys()}")
    executive_summary = callback_context.state.get("executive_summary", {})
    executive_summary_done = executive_summary.get('summary_done', False)
    business_summary = callback_context.state.get("business_summary", {})
    business_summary_done = executive_summary.get('summary_done', False)

    if executive_summary_done and business_summary_done:
        print('Both executive summary and business summary are done. Good for writing agent!')
    else:
        # Option A: raise to stop progression (make sure your runner catches it and stops)
        raise RuntimeError("Pipeline paused: Not both executive summary and business summary are done!")
