import requests
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

import requests
from bs4 import BeautifulSoup, Comment
import json
import re, time
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

def update_agent_state_for_profile(callback_context):
    profile = callback_context.state.get('profile',{})
    if 'detailed_request' in profile:
        callback_context.state["detailed_request"] = profile['detailed_request']
        callback_context.state["email"] = profile['email']
    init_db()
    save_user_profile(profile)
    save_state_after_agent_callback(callback_context)


def save_state_after_agent_callback(callback_context):
    state_dict = callback_context.state.to_dict()
    print(f"[planner_before_agent_callback] Current session state: {state_dict.keys()}")
    try:
        with open(f'agent_state_{callback_context.agent_name}.json', "w") as f:
            json.dump(state_dict, f, indent=2)
    except:
        with open(f"agent_state_{callback_context.agent_name}.txt", "w") as f:
            f.write(str(state_dict))


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
      - final_url: the final_url of the page  
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
            # results.append({
            #     "topic": topic,
            #     "google_title": title,
            #     "url": url,
            #     "uuid": uuid,
            #     "error": "Failed initial GET"
            # })
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
            # results.append({
            #     "topic": topic,
            #     "google_title": title,
            #     "url": url,
            #     "uuid": uuid,
            #     "final_url": final_url,
            #     "canonical_url": '',
            #     "page_title": '',
            #     "summary": '',
            #     "full_text": '',
            #     "error": str(e),
            # })

    return results


def update_agent_state(callback_context: CallbackContext):
    if callback_context.state['plan']:
        callback_context.state["search_queries"] = callback_context.state['plan'].get('search_queries',[])
        prompts = callback_context.state['plan'].get('task_delegation_plan',{})
        callback_context.state['executive_summary_agent_prompt'] = prompts.get('executive_summary_agent','')
        callback_context.state['section_outline'] = callback_context.state['plan'].get('section_outline',{})
    
    profile = callback_context.state.get('profile',{})
    if profile:
        if type(profile) == str:
            profile = json.loads(profile)
        print(callback_context.state['profile'])
        callback_context.state["detailed_request"] = profile['detailed_request']
        callback_context.state["email"] = profile['email']

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

    if executive_summary_done:
        print('Executive summary is done. Good for writing agent!')
    else:
        # Option A: raise to stop progression (make sure your runner catches it and stops)
        raise RuntimeError("Pipeline paused: executive summary is still pending!")


def safe_json_loads(text: str):
    # Remove markdown fences
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    
    # Fix invalid escape sequences by replacing single backslashes
    # not used for valid JSON escapes (",\,/,b,f,n,r,t,u)
    text = re.sub(
        r'\\(?!["\\/bfnrtu])',
        r'\\\\',
        text
    )
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Decode failed:", e)
        print("Trying lenient mode...")

    # secondary attempt: remove control characters
    text = re.sub(r"[\x00-\x1F]+", "", text)

    return json.loads(text)

def prepare_verify_pairs(callback_context: CallbackContext):
    fetch_results = callback_context.state.get('fetch_results_executive','')
    fetch_results = safe_json_loads(fetch_results)
    fetch_uuid = {item['uuid']:item for item  in fetch_results}
    news = callback_context.state.get('newsletter_result')
    pairs = []
    for section in ['executive_summary','business_implications']:
        res = news.get(section,{})
        for item in res:
            uuid = item['uuid']
            full_text = fetch_uuid[uuid].get('full_text','')
            pairs.append({'sentence': item['body'], 'reference': full_text, 'uuid': uuid})
    callback_context.state['verification_pairs'] = pairs
    save_state_after_agent_callback(callback_context)
    # return True

import copy
import re
from typing import List, Dict, Tuple, Any

def after_agent_callback_save_md(callback_context):
    """
    - Parse model output (assumes model returns JSON matching NewsletterOutput)
    - Save JSON to callback_context.state["newsletter_json"]
    - Also write a human-friendly Markdown file newsletter_<session>_<ts>.md
    - Overwrite llm_response text to the JSON string (so ADK UIs show structured output)
    """
    
    out = callback_context.state.get("newsletter_updated")

    if isinstance(out, dict):
        # Build Markdown
        md_lines = []
        md_lines.append(f"# {out['newsletter_title']}\n")
        md_lines.append(f"*Date: {out.get('date')}*\n")
        md_lines.append(f"**{out.get('short_blurb')}**\n")

        md_lines.append("## Executive Summary\n")
        for item in out.get('executive_summary'):
            md_lines.append(f"### {item.get('heading')}\n")
            md_lines.append(item.get('body') + "\n")

        md_lines.append("## Business & Industry Insights\n")
        for item in out.get('business_implications'):
            md_lines.append(f"### {item.get('heading')}\n")
            md_lines.append(item.get('body') + "\n")

        md_lines.append("## TL;DR\n")
        md_lines.append(out.get('tl_dr') + "\n")

        md_lines.append("## Citations / Sources\n")
        for c in out.get('citations'):
            md_lines.append(f"- {c}")

        md_lines.append("\n## Call to Action\n")
        md_lines.append(out.get('call_to_action') + "\n")

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

        callback_context.state["newsletter_md_text"] = md_text


def apply_verification_updates(callback_context: CallbackContext,
    section_keys_with_items: List[str] = None
):
    """
    Replace inaccurate sentences in a newsletter dict using verification results.
    Matching is done by (uuid, sentence) combination:
      - For each verification entry where accuracy_or_not is False,
        locate the newsletter section item with the same uuid and replace the
        first exact occurrence of the sentence in that item's 'body' with the
        provided 'modified_version'.

    Args:
      newsletter: newsletter dict (same shape you provided).
      verifications: list of verification dicts, each with keys:
          'sentence' (str), 'uuid' (str or id), 'accuracy_or_not' (bool),
          'modified_version' (str)
      section_keys_with_items: optional list of keys in newsletter that contain
          list-of-dict items (defaults to common keys).

    Returns:
      (updated_newsletter, changes)
        - updated_newsletter: deep-copied newsletter with replacements applied
        - changes: list of dicts {uuid, original_sentence, modified_version, section_key, item_index, replaced}
    """
    newsletter = callback_context.state.get('newsletter_result',{})
    verification_result = callback_context.state.get('verification_result',{})
    if 'VerificationOutput' in verification_result:
        verifications = verification_result['VerificationOutput']

    if section_keys_with_items is None:
        section_keys_with_items = [
            "executive_summary",
            "business_implications",
        ]

    updated = copy.deepcopy(newsletter)
    changes = []

    # Build map uuid -> list of (section_key, index, item_ref) for fast lookup
    uuid_index = {}
    for key in section_keys_with_items:
        items = updated.get(key)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            u = item.get("uuid") or item.get("id") or item.get("doc_id") or item.get("uuid_str")
            if not u:
                continue
            uuid_index.setdefault(str(u), []).append((key, idx, item))

    # Helper to perform a safe single-sentence replacement in a larger body string
    def replace_sentence_in_body(body: str, sentence: str, replacement: str) -> Tuple[str, bool]:
        """
        Replace the first exact match of 'sentence' inside 'body' with 'replacement'.
        Returns (new_body, replaced_flag).
        Uses a regex to find the sentence as a substring, allowing surrounding whitespace.
        """
        if not body or not sentence:
            return body, False

        # escape sentence for regex, but consider potential whitespace differences:
        s_esc = re.escape(sentence.strip())
        # look for the sentence followed by punctuation/space or end of string
        pattern = re.compile(r"(?<!\S)" + s_esc + r"(?!\S)", flags=re.MULTILINE)
        # Try simple literal replace first (fast)
        if sentence in body:
            new_body = body.replace(sentence, replacement, 1)
            return new_body, True

        # Fallback to regex search (less strict: match normalized whitespace)
        m = pattern.search(body)
        if m:
            start, end = m.span()
            new_body = body[:start] + replacement + body[end:]
            return new_body, True

        return body, False

    # Process each verification entry
    for v in verifications:
        sentence = v.get("sentence")
        uuid = v.get("uuid")
        accurate = v.get("accuracy_or_not", True)
        modified = v.get("modified_version", sentence)

        if not sentence or uuid is None:
            # skip entries that don't have both sentence and uuid
            continue

        uid = str(uuid)
        matched = False

        # 1) Try to find by uuid in the prebuilt index
        locations = uuid_index.get(uid, [])
        for (section_key, idx, item) in locations:
            body = item.get("body") or item.get("summary") or item.get("text") or ""
            if not isinstance(body, str):
                continue
            if not accurate:
                new_body, replaced = replace_sentence_in_body(body, sentence, modified)
                if replaced:
                    # commit change into updated newsletter
                    updated[section_key][idx]['body'] = new_body
                    changes.append({
                        "uuid": uid,
                        "original_sentence": sentence,
                        "modified_version": modified,
                        "section_key": section_key,
                        "item_index": idx,
                        "replaced": True
                    })
                    matched = True
                    break  # stop searching other items with same uuid
                else:
                    # record attempted but not found in this item's body
                    # (we may try other locations)
                    continue
            else:
                # accurate -> no change, but record that we found matching location
                if sentence in body:
                    changes.append({
                        "uuid": uid,
                        "original_sentence": sentence,
                        "modified_version": sentence,
                        "section_key": section_key,
                        "item_index": idx,
                        "replaced": False,
                        "note": "verified accurate; no change"
                    })
                    matched = True
                    break

        if matched:
            continue

        # 2) If not matched by uuid index, try scanning other likely places
        # for items with same uuid in any list (fallback)
        for key, items in updated.items():
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                u = item.get("uuid") or item.get("id") or item.get("doc_id") or item.get("uuid_str")
                if str(u) != uid:
                    continue
                body = item.get("body") or item.get("summary") or item.get("text") or ""
                if not isinstance(body, str):
                    continue
                if not accurate:
                    new_body, replaced = replace_sentence_in_body(body, sentence, modified)
                    if replaced:
                        updated[key][idx]['body'] = new_body
                        changes.append({
                            "uuid": uid,
                            "original_sentence": sentence,
                            "modified_version": modified,
                            "section_key": key,
                            "item_index": idx,
                            "replaced": True,
                            "fallback": True
                        })
                        matched = True
                        break
            if matched:
                break
        if matched:
            continue

        # 3) Final fallback: replace anywhere in the entire newsletter (best-effort)
        if not accurate:
            replaced_any = False
            for key, val in updated.items():
                # handle top-level string fields
                if isinstance(val, str):
                    new_val, replaced = replace_sentence_in_body(val, sentence, modified)
                    if replaced:
                        updated[key] = new_val
                        replaced_any = True
                        changes.append({
                            "uuid": uid,
                            "original_sentence": sentence,
                            "modified_version": modified,
                            "section_key": key,
                            "item_index": None,
                            "replaced": True,
                            "fallback": "global"
                        })
                        break
                # handle lists of strings
                if isinstance(val, list):
                    for idx, elem in enumerate(val):
                        if isinstance(elem, str):
                            new_elem, replaced = replace_sentence_in_body(elem, sentence, modified)
                            if replaced:
                                updated[key][idx] = new_elem
                                replaced_any = True
                                changes.append({
                                    "uuid": uid,
                                    "original_sentence": sentence,
                                    "modified_version": modified,
                                    "section_key": key,
                                    "item_index": idx,
                                    "replaced": True,
                                    "fallback": "global_list"
                                })
                                break
                        elif isinstance(elem, dict):
                            body = elem.get("body") or elem.get("summary") or elem.get("text")
                            if isinstance(body, str):
                                new_body, replaced = replace_sentence_in_body(body, sentence, modified)
                                if replaced:
                                    updated[key][idx]['body'] = new_body
                                    replaced_any = True
                                    changes.append({
                                        "uuid": uid,
                                        "original_sentence": sentence,
                                        "modified_version": modified,
                                        "section_key": key,
                                        "item_index": idx,
                                        "replaced": True,
                                        "fallback": "global_list_dict"
                                    })
                                    break
                    if replaced_any:
                        break
                if replaced_any:
                    break
            if not replaced_any:
                changes.append({
                    "uuid": uid,
                    "original_sentence": sentence,
                    "modified_version": modified,
                    "section_key": None,
                    "item_index": None,
                    "replaced": False,
                    "note": "could not find sentence to replace"
                })
        else:
            # accurate but not matched — record as unchecked/unmatched
            changes.append({
                "uuid": uid,
                "original_sentence": sentence,
                "modified_version": sentence,
                "section_key": None,
                "item_index": None,
                "replaced": False,
                "note": "accurate but not matched to any section"
            })
    callback_context.state['newsletter_updated'] = updated
    callback_context.state['newsletter_changes'] = changes

    after_agent_callback_save_md(callback_context)
    save_state_after_agent_callback(callback_context)


def save_search_results(callback_context: CallbackContext):
    search_results_executive = callback_context.state.get('search_results_executive','')
    try:
        search_results = safe_json_loads(search_results_executive)
    except:
        search_results = []
    email = callback_context.state.get('email','')
    for article in search_results:
        save_article_for_user(email, article)
    save_state_after_agent_callback(callback_context)


def create_uuid_for_search_results(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse:

    resp_text = ""
    if llm_response.content and llm_response.content.parts:
        resp_text = llm_response.content.parts[0].text

    parsed = None
    try:
        # Agent should output JSON; parse it.
        parsed = safe_json_loads(resp_text)
    except Exception:
        # Try to fall back to state if agent already wrote structured data there:
        parsed = callback_context.state.get("search_results_executive")
    if not parsed:
        return llm_response
    import uuid
    for item in parsed:
        item['uuid'] = str(uuid.uuid4())
    llm_response.content.parts[0].text = json.dumps(parsed)
    callback_context.state['search_results_executive'] = json.dumps(parsed)
    return llm_response


import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List

DB_PATH = Path(__file__).parent / "ai_newsletter.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # user profiles
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        profile_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # article index
    cur.execute("""
    CREATE TABLE IF NOT EXISTS article_index (
        user_email TEXT,
        article_id TEXT,
        url TEXT,
        published_at TEXT,
        title TEXT,
        summary TEXT,
        PRIMARY KEY (user_email, article_id)
    );
    """)

    conn.commit()
    conn.close()
    print("✅ SQLite initialized at:", DB_PATH)

def get_db():
    return sqlite3.connect(DB_PATH)

# ---------------- User Profiles ----------------

def load_user_profile(email: str) -> Dict[str, Any]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT profile_json FROM users WHERE email = ?",
            (email,),
        )
        row = cur.fetchone()
    if row:
        return json.loads(row[0])
    return {"email": email}

def save_user_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    email = profile.get("email")
    if not email:
        raise ValueError("Profile must contain 'email'")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO users (email, profile_json)
            VALUES (?, ?)
            ON CONFLICT(email) DO UPDATE SET 
                profile_json = excluded.profile_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (email, json.dumps(profile)),
        )
    return {"ok": True}

# ---------------- Article Index ----------------

def get_seen_article_ids(user_email: str) -> List[str]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT article_id FROM article_index WHERE user_email = ?",
            (user_email,),
        )
        return [row[0] for row in cur.fetchall()]

def save_article_for_user(user_email: str, article: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO article_index
            (user_email, article_id, url, published_at, title, summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_email,
                article["uuid"],
                article["url"],
                article.get("publish_date"),
                article.get("title"),
                article.get("short_summary", None),
            ),
        )


def convert_newsletter_json_to_html(data: dict) -> str:
    """
    Convert the summarized newsletter JSON into a full HTML email.
    All URLs in sections are replaced with a “Read More” hyperlink.
    """

    def section_list_to_html(sections):
        html = ""
        for sec in sections:
            read_more = f'<p><a href="{sec["final_url"]}" target="_blank">Read More →</a></p>'
            html += f"""
            <div style="margin-bottom: 28px;">
                <h3 style="margin: 0 0 8px 0; color: #0d6efd; font-size: 20px;">
                    {sec['heading']}
                </h3>
                <p style="margin: 0 0 8px 0; line-height: 1.55;">
                    {sec['body']}
                </p>
                {read_more}
            </div>
            """
        return html

    executive_html = section_list_to_html(data.get("executive_summary",[]))
    business_html = section_list_to_html(data.get("business_implications",[]))

    citations_html = "".join(
        f'<li><a href="{c}" target="_blank">{c}</a></li>' for c in data.get("citations",[])
    )

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; background: #f7f7f7; padding: 20px;">
        <table style="max-width: 700px; margin: auto; background: white; padding: 32px; border-radius: 12px;">
            <tr>
                <td>

                    <h1 style="font-size: 28px; margin-bottom: 4px;">
                        {data['newsletter_title']}
                    </h1>
                    <p style="color: gray; margin-top: 0;">{data['date']}</p>

                    <p style="font-size: 17px; line-height: 1.6;">
                        {data['short_blurb']}
                    </p>

                    <hr style="margin: 28px 0;" />

                    <h2 style="font-size: 24px; margin-bottom: 16px;">Executive Summary</h2>
                    {executive_html}

                    <hr style="margin: 28px 0;" />

                    <h2 style="font-size: 24px; margin-bottom: 16px;">Business Implications</h2>
                    {business_html}

                    <hr style="margin: 28px 0;" />

                    <h2 style="font-size: 24px; margin-bottom: 16px;">Citations</h2>
                    <ul style="line-height: 1.6; padding-left: 20px;">
                        {citations_html}
                    </ul>

                    <hr style="margin: 28px 0;" />

                    <h2 style="font-size: 24px; margin-bottom: 16px;">TL;DR</h2>
                    <p style="white-space: pre-line; line-height: 1.55;">
                        {data['tl_dr']}
                    </p>

                    <hr style="margin: 28px 0;" />

                    <h2 style="font-size: 24px; margin-bottom: 16px;">Call to Action</h2>
                    <p style="line-height: 1.55;">
                        {data.get('call_to_action','')}
                    </p>

                </td>
            </tr>
        </table>
    </body>
    </html>
    """

import os
import smtplib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
import unicodedata
import logging

logger = logging.getLogger(__name__)

def _sanitize_html_for_email(html: str) -> str:
    """
    Normalize and sanitize HTML so common unicode issues (NBSP, weird control chars)
    don't break ASCII-only fallbacks or legacy send paths.
    """
    if html is None:
        return ""
    # Normalize to composed form
    html = unicodedata.normalize("NFC", html)
    # Replace common problematic characters: NBSP -> normal space
    html = html.replace("\u00A0", " ")
    # Optionally strip other non-printable controls except newline/tab
    html = "".join(ch for ch in html if (ch == "\n" or ch == "\t" or (32 <= ord(ch) <= 0x10FFFF)))
    return html

def send_newsletter_email(to_email: str, subject: str, newsletter_json: dict) -> dict:
    """
    Send an HTML newsletter email using SMTP, or simulate sending in demo mode.

    This function takes structured newsletter data (`newsletter_json`), converts it
    into HTML, sanitizes the HTML for email compatibility, and then sends it using
    SMTP. If demo mode is enabled, the email is NOT sent; instead, a debug file is
    written to `./debug_newsletters/` and metadata is returned.

    Parameters
    ----------
    to_email : str
        Recipient email address, inferred from context state.

    subject : str
        Subject line for the email, get the title from newsletter_json.

    newsletter_json : dict
        A structured newsletter dictionary (typically containing title, sections,
        articles, summaries, URLs, timestamps, etc.) that will be converted to HTML
        using `convert_newsletter_json_to_html()`.

    Environment Variables
    ---------------------
    NEWSLETTER_DEMO_MODE : str
        "1", "true", or "yes" enables demo mode.
        In demo mode:
            - No SMTP connection is attempted.
            - HTML is generated and sanitized.
            - HTML output is written to ./debug_newsletters/newsletter_<timestamp>.html
            - A `mock_sent` status dictionary is returned.

    SMTP_HOST : str
        Hostname of the SMTP server (e.g., "smtp.gmail.com").

    SMTP_PORT : str or int
        SMTP port (typically 587 for TLS).

    SMTP_USER : str
        Username/email for SMTP authentication. **Must be ASCII-only** or the SMTP
        library may throw UnicodeEncodeError during authentication.

    SMTP_PASS : str
        Password or app password. **Must be ASCII-only**. If non-ASCII characters
        such as NBSP (`\\u00A0`) or zero-width characters are present, SMTP auth
        will fail before the email can be sent.

    NEWSLETTER_FROM_EMAIL : str (optional)
        Overrides the "From" header. Defaults to SMTP_USER if not provided.

    Behavior
    --------
    1. Convert newsletter JSON → HTML via `convert_newsletter_json_to_html()`.
    2. Sanitize HTML by:
        - Normalizing Unicode (NFC)
        - Replacing non-breaking spaces (`\\u00A0`) with normal spaces
        - Removing invisible Unicode control characters
    3. If demo mode:
        - Write HTML to a timestamped file
        - Return { "status": "mock_sent", "debug_file": "<path>" }
    4. If SMTP mode:
        - Validate required SMTP environment variables
        - Build a MIME EmailMessage with both text/plain and text/html parts
        - Connect to SMTP with TLS, authenticate, and send the message

    Returns
    -------
    dict
        A structured status object. Possible return formats:

        Successful send:
            {
                "status": "sent",
                "message_id": "smtp-send_message"
            }

        Demo mode (email not sent):
            {
                "status": "mock_sent",
                "message_id": "demo-<timestamp>",
                "debug_file": "debug_newsletters/newsletter_<timestamp>.html"
            }

        Missing SMTP configuration:
            {
                "status": "config_error",
                "message": "SMTP config missing; check environment variables."
            }

        SMTP authentication error:
            {
                "status": "auth_error",
                "message": "SMTP authentication failed: <error>"
            }

        Unicode encoding error (most common if credentials contain non-ASCII):
            {
                "status": "encoding_error",
                "message": "Unicode encoding failed: <error>"
            }

        Generic error:
            {
                "status": "error",
                "message": "SMTP error: <exception repr>"
            }

    Notes
    -----
    - Even if your HTML content is fully UTF-8, **SMTP authentication still requires
      ASCII-only credentials**. Characters like non-breaking spaces (`\\u00A0`) often
      appear in secrets when copied from password managers or webpages.
    - Sanitizing HTML does NOT modify subject, SMTP_USER or SMTP_PASS; those must be
      clean in the environment before calling this function.
    - Debug mode is strongly recommended during development to avoid accidental sends.

    """

    
    demo_mode = os.getenv("NEWSLETTER_DEMO_MODE", "0").lower() in ("1", "true", "yes")

    # sanitize HTML before anything else
    html = convert_newsletter_json_to_html(newsletter_json)
    try:
        html_safe = _sanitize_html_for_email(html)
        # html_safe = html
    except Exception as e:
        logger.exception("Failed to sanitize HTML")
        html_safe = html or ""

    # -----------------------
    # DEMO MODE (no real SMTP)
    # -----------------------
    if demo_mode:
        try:
            print("=== [DEMO] send_newsletter_email called ===")
            print("To:", to_email)
            print("Subject:", subject)

            debug_dir = Path("debug_newsletters")
            debug_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = debug_dir / f"newsletter_{ts}.html"
            fname.write_text(html_safe, encoding="utf-8")
            print(f"=== [DEMO] Saved newsletter HTML to {fname}")

            return {
                "status": "mock_sent",
                "message_id": f"demo-{ts}",
                "debug_file": str(fname),
            }
        except Exception as e:
            logger.exception("Demo-mode write failed")
            return {"status": "error", "message": f"demo_write_error: {e}"}

    # -----------------------
    # REAL SMTP MODE
    # -----------------------
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_email = os.getenv("NEWSLETTER_FROM_EMAIL", user)

    if not all([host, port, user, password, from_email]):
        # Return config error rather than raising
        return {
            "status": "config_error",
            "message": "SMTP config missing; check .env (SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/NEWSLETTER_FROM_EMAIL).",
        }

    # Build EmailMessage (handles charset/encoding properly)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    # Plain-text fallback
    plain_text = "This message contains HTML. If you cannot see it, please view in a browser."
    msg.set_content(plain_text)

    # Attach the HTML alternative. EmailMessage will use utf-8 as needed.
    msg.add_alternative(html_safe, subtype="html")

    try:
        with smtplib.SMTP(host, port, timeout=60) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        return {"status": "sent", "message_id": "smtp-send_message"}
    except smtplib.SMTPAuthenticationError as e:
        logger.exception("SMTP auth error")
        return {"status": "auth_error", "message": f"SMTP authentication failed: {e}"}
    except UnicodeEncodeError as e:
        # Capture and return a helpful message if encoding still fails
        logger.exception("UnicodeEncodeError during SMTP send")
        return {"status": "encoding_error", "message": f"Unicode encoding failed: {e}"}
    except Exception as e:
        logger.exception("SMTP error")
        # include repr(e) so the caller sees the original message (e.g., your 'ascii' codec error)
        return {"status": "error", "message": f"SMTP error: {e!r}"}
