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

def format_output(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> LlmResponse:
    # 1. Inspect or modify the response if needed
    # (optional) For example, you could log or sanitize:
    resp_text = ""
    if llm_response.content and llm_response.content.parts:
        resp_text = llm_response.content.parts[0].text
        if resp_text:
            if '```json' in resp_text:
                try:
                    results_json = json.loads(resp_text.split('```json')[-1].replace('```',''))
                    llm_response.content.parts[0].text = json.dumps(results_json)
                    json.dump(results_json,open(f'agent_state_{callback_context.agent_name}.json','w'))
                except:
                    with open(f"agent_state_{callback_context.agent_name}.txt", "w") as f:
                        f.write(resp_text)
            else:
                with open(f"agent_state_{callback_context.agent_name}.txt", "w") as f:
                    f.write(resp_text)
    return llm_response

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

# def 
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
    # business_summary = callback_context.state.get("business_summary", {})
    # business_summary_done = executive_summary.get('summary_done', False)

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