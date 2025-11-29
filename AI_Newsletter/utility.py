import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup, Comment
import copy
from google.adk.agents.callback_context import CallbackContext

import sqlite3

from google.adk.models import LlmResponse  # correct for your ADK version


# ----------------------------
#  HTML parsing / content utils
# ----------------------------

CONTENT_HINTS = [
    "article", "main", "content", "post", "entry", "body-content",
    "post-content", "article-body", "main-content", "StoryBodyCompanion",
    "Section1", "RichText", "td-post-content",
]



def safe_json_loads(text: str) -> Any:
    """
    More lenient JSON loader:
      - strips ```json fences
      - fixes some unsafe backslashes
      - strips control chars if first parse fails
    """
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)

    text = re.sub(
        r'\\(?!["\\/bfnrtu])',
        r"\\\\",
        text,
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Decode failed:", e)
        print("Trying lenient mode...")

    text = re.sub(r"[\x00-\x1F]+", "", text)
    return json.loads(text)

def save_state_after_agent_callback(callback_context: CallbackContext) -> None:
    """Persist the current state to a JSON (or TXT) file for debugging."""
    state_dict = callback_context.state.to_dict()
    print(f"[save_state_after_agent_callback] Current session state keys: {state_dict.keys()}")

    filename_base = f"agent_state_{callback_context.agent_name}"
    try:
        with open(f"{filename_base}.json", "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    except Exception:
        with open(f"{filename_base}.txt", "w", encoding="utf-8") as f:
            f.write(str(state_dict))


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
                callback_context.state["formatted_questions"] = [
                    f"Question {q}\n " for q in questions
                ]

def update_agent_state_for_profile(callback_context: CallbackContext) -> None:
    """
    Normalize `state['profile']` to a dict and expose detailed_request in state.
    This is used as an after_agent_callback for the profile agent.
    """
    profile_raw = callback_context.state.get("profile")
    if profile_raw is None:
        print("[profile_callback] No 'profile' key found in state")
        return

    if isinstance(profile_raw, dict):
        profile_data = profile_raw
    else:
        try:
            profile_data = safe_json_loads(profile_raw)
        except Exception as e:
            print(f"Error parsing JSON from profile_agent: {e}")
            print(f"Raw output: {profile_raw}")
            return

    callback_context.state["profile"] = profile_data
    callback_context.state["email"] = profile_data.get('email','example@example.com')
    callback_context.state["detailed_request"] = profile_data.get(
        "detailed_request", "No request found"
    )

    init_db()
    save_user_profile(profile_data)
    save_state_after_agent_callback(callback_context)

def update_agent_state_for_recommender(callback_context: CallbackContext) -> None:
    """
    Normalize `state['profile']` to a dict and expose detailed_request in state.
    This is used as an after_agent_callback for the profile agent.
    """
    refined_topics = callback_context.state.get("refined_topics")
    if refined_topics is None:
        print("[profile_callback] No 'refined_topics' key found in state")
        return

    if isinstance(refined_topics, dict):
        updated_request = refined_topics.get('detailed_request_updated','')
    else:
        try:
            updated_request = safe_json_loads(refined_topics).get('detailed_request_updated','')
        except Exception as e:
            print(f"Error parsing JSON from refined_topics: {e}")
            print(f"Raw output: {refined_topics}")
            return

    callback_context.state["profile"]['detailed_request'] = updated_request
    callback_context.state["detailed_request"] =updated_request

    save_state_after_agent_callback(callback_context)



def update_agent_state_planner(callback_context: CallbackContext):
    if callback_context.state['plan']:
        callback_context.state["search_queries"] = callback_context.state['plan'].get('search_queries',[])
        prompts = callback_context.state['plan'].get('task_delegation_plan',{})
        callback_context.state['executive_summary_agent_prompt'] = prompts.get('executive_summary_agent','')
        callback_context.state['section_outline'] = callback_context.state['plan'].get('section_outline',{})
    
    save_state_after_agent_callback(callback_context)


# for fetch web pages

def clean_text(txt: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", txt).strip()


def extract_main_content(soup: BeautifulSoup, page_title: str | None = None) -> str:
    """Extract main article-like content from HTML soup."""
    # Remove noisy tags/sections
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    body = soup.body or soup

    # Try structural candidates first
    candidates: List[Tuple[int, str]] = []
    for tag in body.find_all():
        id_class = " ".join((tag.get("id") or "").split() + (tag.get("class") or []))
        if any(hint.lower() in id_class.lower() for hint in CONTENT_HINTS):
            txt = clean_text(tag.get_text(" ", strip=True))
            if len(txt) > 300:
                candidates.append((len(txt), txt))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Fallback: text-dense blocks
    blocks: List[Tuple[int, str]] = []
    for tag in body.find_all(["div", "section", "article", "p"]):
        txt = clean_text(tag.get_text(" ", strip=True))
        if len(txt) > 200:
            blocks.append((len(txt), txt))

    if blocks:
        blocks.sort(reverse=True)
        best = blocks[0][1]
        if page_title:
            keywords = [w.lower() for w in page_title.split() if len(w) > 4]
            filtered = "\n".join(
                p for p in best.split(". ") if any(k in p.lower() for k in keywords)
            )
            if len(filtered) > 0.3 * len(best):
                return filtered
        return best

    return clean_text(body.get_text(" ", strip=True))


def fetch_page_details(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fetch detailed information for a list of web pages.

    Each `page` dict is expected to contain:
      - topic
      - title
      - url
      - uuid
    this function issues HTTP GET
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
    results: List[Dict[str, Any]] = []

    for page in pages:
        topic = page.get("topic")
        title = page.get("title")
        url = page.get("url")
        page_uuid = page.get("uuid")

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

            canonical_tag = soup.find("link", rel="canonical")
            canonical_url = (
                canonical_tag["href"]
                if canonical_tag is not None and canonical_tag.has_attr("href")
                else None
            )

            page_title = soup.title.string.strip() if soup.title else None
            main_text = extract_main_content(soup, page_title=page_title)
            summary = main_text[:1000]

            results.append(
                {
                    "topic": topic,
                    "google_title": title,
                    "url": url,
                    "uuid": page_uuid,
                    "final_url": final_url,
                    "canonical_url": canonical_url,
                    "page_title": page_title,
                    "summary": summary,
                    "full_text": main_text,
                }
            )
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


def planner_before_agent_callback(callback_context: CallbackContext) -> None:
    """
    Guard before planner agent: ensure clarifications are done.
    Raises RuntimeError to pause pipeline if requirement step not completed.
    """
    state_dict = callback_context.state.to_dict()
    print(
        f"[planner_before_agent_callback] Current session state: {state_dict.keys()}"
    )

    clarifications = callback_context.state.get("request_clarification", {})
    request_clarification_done = clarifications.get("request_clarification_done", False)

    if not clarifications or not request_clarification_done:
        raise RuntimeError("Pipeline paused: requirement_agent needs clarifications")
    else:
        callback_context.state["detailed_request"] = clarifications.get(
            "detailed_request", ""
        )

def writer_before_agent_callback(callback_context):
    """
    Guard before writer agent: ensure executive summary is done.
    """
    state_dict = callback_context.state.to_dict()
    print(
        f"[writer_before_agent_callback] Current session state: {state_dict.keys()}"
    )

    executive_summary = callback_context.state.get("executive_summary", {})
    executive_summary_done = executive_summary.get('summary_done', False)

    if executive_summary_done:
        print("Executive summary is done. Good for writing agent!")
    else:
        raise RuntimeError("Pipeline paused: executive summary is still pending!")



def prepare_verify_pairs(callback_context: CallbackContext) -> None:
    """Build (sentence, reference, uuid) pairs for verification."""
    fetch_results_raw = callback_context.state.get("fetch_results_executive", "")
    fetch_results = safe_json_loads(fetch_results_raw)
    fetch_by_uuid = {item["uuid"]: item for item in fetch_results}

    news = callback_context.state.get("newsletter_result", {})
    pairs: List[Dict[str, Any]] = []

# utility.py
def prepare_verify_pairs(callback_context):
    state = callback_context.state

    newsletter = state.get("newsletter_result") or {}
    exec_summary = state.get("executive_summary") or {}

    # Adjust to your actual schemas
    # Assume newsletter_result["executive_summary"] is a list of items
    # and exec_summary["items"] (or similar) is the list of article summaries.

    newsletter_items = newsletter.get("executive_summary", []) or []
    source_items = exec_summary.get("items", []) or exec_summary or []

    # Build a simple uuid -> summary text mapping
    source_by_uuid = {}
    if isinstance(source_items, list):
        for s in source_items:
            if isinstance(s, dict):
                uid = str(s.get("uuid") or "")
                if uid:
                    # choose the key that holds the dense summary
                    source_by_uuid[uid] = s.get("summary") or ""

    verification_pairs = []

    for item in newsletter_items:
        if not isinstance(item, dict):
            continue
        uuid = str(item.get("uuid") or "")
        sentence = item.get("body") or item.get("heading") or ""
        if not uuid or not sentence:
            continue

        reference = source_by_uuid.get(uuid, "")
        if not reference:
            # fallback to body itself if we have nothing
            reference = item.get("body", "")
        if not reference:
            continue

        verification_pairs.append(
            {
                "uuid": uuid,
                "sentence": sentence,
                "reference": reference,
            }
        )

    state["verification_pairs"] = verification_pairs
    return None


def after_agent_callback_save_md(callback_context: CallbackContext) -> None:
    """
    Build a Markdown newsletter from `state['newsletter_updated']`
    and write it to disk, plus store the MD text in state.
    """
    out = callback_context.state.get("newsletter_updated")

    if not isinstance(out, dict):
        return

    md_lines: List[str] = []

    md_lines.append(f"# {out.get('newsletter_title', '')}\n")
    md_lines.append(f"*Date: {out.get('date', '')}*\n")
    md_lines.append(f"**{out.get('short_blurb', '')}**\n")

    md_lines.append("## Executive Summary\n")
    for item in out.get("executive_summary", []):
        md_lines.append(f"### {item.get('heading', '')}\n")
        md_lines.append((item.get("body") or "") + "\n")

    md_lines.append("## Business & Industry Insights\n")
    for item in out.get("business_implications", []):
        md_lines.append(f"### {item.get('heading', '')}\n")
        md_lines.append((item.get("body") or "") + "\n")

    md_lines.append("## TL;DR\n")
    md_lines.append((out.get("tl_dr") or "") + "\n")

    md_lines.append("## Citations / Sources\n")
    for c in out.get("citations", []):
        md_lines.append(f"- {c}")

    md_lines.append("\n## Call to Action\n")
    md_lines.append((out.get("call_to_action") or "") + "\n")

    md_text = "\n".join(md_lines)

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


def apply_verification_updates(
    callback_context: CallbackContext,
    section_keys_with_items: List[str] | None = None,
) -> None:
    """
    Apply verification result to newsletter: replace inaccurate sentences
    using 'modified_version' where accuracy_or_not is False.
    """
    import copy as _copy

    newsletter = callback_context.state.get("newsletter_result", {})
    verification_result = callback_context.state.get("verification_result", {})

    verifications = verification_result.get("VerificationOutput", [])

    if section_keys_with_items is None:
        section_keys_with_items = [
            "executive_summary",
            "business_implications",
        ]

    updated = _copy.deepcopy(newsletter)
    changes: List[Dict[str, Any]] = []

    def replace_sentence_in_body(body: str, sentence: str, replacement: str) -> Tuple[str, bool]:
        if not body or not sentence:
            return body, False

        if sentence in body:
            return body.replace(sentence, replacement, 1), True

        s_esc = re.escape(sentence.strip())
        pattern = re.compile(r"(?<!\S)" + s_esc + r"(?!\S)", flags=re.MULTILINE)
        m = pattern.search(body)
        if m:
            start, end = m.span()
            return body[:start] + replacement + body[end:], True

        return body, False

    uuid_index: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
    for key in section_keys_with_items:
        items = updated.get(key)
        if not isinstance(items, List):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            u = (
                item.get("uuid")
                or item.get("id")
                or item.get("doc_id")
                or item.get("uuid_str")
            )
            if not u:
                continue
            uuid_index.setdefault(str(u), []).append((key, idx, item))

    for v in verifications:
        sentence = v.get("sentence")
        uid = v.get("uuid")
        accurate = v.get("accuracy_or_not", True)
        modified = v.get("modified_version", sentence)

        if not sentence or uid is None:
            continue

        uid_str = str(uid)
        matched = False

        # 1) Try direct uuid index
        for (section_key, idx, item) in uuid_index.get(uid_str, []):
            body = (
                item.get("body")
                or item.get("summary")
                or item.get("text")
                or ""
            )
            if not isinstance(body, str):
                continue

            if not accurate:
                new_body, replaced = replace_sentence_in_body(body, sentence, modified)
                if replaced:
                    updated[section_key][idx]["body"] = new_body
                    changes.append(
                        {
                            "uuid": uid_str,
                            "original_sentence": sentence,
                            "modified_version": modified,
                            "section_key": section_key,
                            "item_index": idx,
                            "replaced": True,
                        }
                    )
                    matched = True
                    break
            else:
                if sentence in body:
                    changes.append(
                        {
                            "uuid": uid_str,
                            "original_sentence": sentence,
                            "modified_version": sentence,
                            "section_key": section_key,
                            "item_index": idx,
                            "replaced": False,
                            "note": "verified accurate; no change",
                        }
                    )
                    matched = True
                    break

        if matched:
            continue

        # 2) Fallback: search all sections
        if not accurate:
            replaced_any = False
            for key, val in updated.items():
                if isinstance(val, str):
                    new_val, replaced = replace_sentence_in_body(val, sentence, modified)
                    if replaced:
                        updated[key] = new_val
                        replaced_any = True
                        changes.append(
                            {
                                "uuid": uid_str,
                                "original_sentence": sentence,
                                "modified_version": modified,
                                "section_key": key,
                                "item_index": None,
                                "replaced": True,
                                "fallback": "global",
                            }
                        )
                        break
                if isinstance(val, list):
                    for idx, elem in enumerate(val):
                        if isinstance(elem, str):
                            new_elem, replaced = replace_sentence_in_body(
                                elem, sentence, modified
                            )
                            if replaced:
                                updated[key][idx] = new_elem
                                replaced_any = True
                                changes.append(
                                    {
                                        "uuid": uid_str,
                                        "original_sentence": sentence,
                                        "modified_version": modified,
                                        "section_key": key,
                                        "item_index": idx,
                                        "replaced": True,
                                        "fallback": "global_list",
                                    }
                                )
                                break
                        elif isinstance(elem, dict):
                            body = (
                                elem.get("body")
                                or elem.get("summary")
                                or elem.get("text")
                            )
                            if isinstance(body, str):
                                new_body, replaced = replace_sentence_in_body(
                                    body, sentence, modified
                                )
                                if replaced:
                                    updated[key][idx]["body"] = new_body
                                    replaced_any = True
                                    changes.append(
                                        {
                                            "uuid": uid_str,
                                            "original_sentence": sentence,
                                            "modified_version": modified,
                                            "section_key": key,
                                            "item_index": idx,
                                            "replaced": True,
                                            "fallback": "global_list_dict",
                                        }
                                    )
                                    break
                    if replaced_any:
                        break
                if replaced_any:
                    break

            if not replaced_any:
                changes.append(
                    {
                        "uuid": uid_str,
                        "original_sentence": sentence,
                        "modified_version": modified,
                        "section_key": None,
                        "item_index": None,
                        "replaced": False,
                        "note": "could not find sentence to replace",
                    }
                )
        else:
            changes.append(
                {
                    "uuid": uid_str,
                    "original_sentence": sentence,
                    "modified_version": sentence,
                    "section_key": None,
                    "item_index": None,
                    "replaced": False,
                    "note": "accurate but not matched to any section",
                }
            )

    callback_context.state["newsletter_updated"] = updated
    callback_context.state["newsletter_changes"] = changes

    after_agent_callback_save_md(callback_context)
    save_state_after_agent_callback(callback_context)


def save_search_results(callback_context: CallbackContext) -> None:
    """Persist search results into the DB using save_article_for_user."""
    raw = callback_context.state.get("search_results_executive", "")
    try:
        search_results = safe_json_loads(raw)
    except Exception:
        search_results = []

    email = callback_context.state.get("email", "")
    for article in search_results:
        save_article_for_user(email, article)

    save_state_after_agent_callback(callback_context)


def create_uuid_for_search_results(
    callback_context: CallbackContext | None = None,
    llm_response: LlmResponse | None = None,
    **kwargs: Any,
) -> LlmResponse | None:
    """
    After-model callback for the search-results agent.

    ADK calls this as:
        create_uuid_for_search_results(callback_context=..., llm_response=...)

    We:
      - read the JSON text from the LLM response
      - add a 'uuid' field to each item
      - write it back into the response and into state["search_results_executive"]
    """
    if llm_response is None:
        return None

    resp_text = ""
    if llm_response.content and llm_response.content.parts:
        resp_text = llm_response.content.parts[0].text or ""

    if not resp_text.strip():
        return llm_response

    parsed: Any = None
    try:
        parsed = safe_json_loads(resp_text)
    except Exception:
        if callback_context is not None:
            parsed = callback_context.state.get("search_results_executive")

    if not parsed:
        return llm_response

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return llm_response

    for item in parsed:
        if isinstance(item, dict):
            item["uuid"] = str(uuid.uuid4())

    new_text = json.dumps(parsed, ensure_ascii=False)

    if llm_response.content and llm_response.content.parts:
        llm_response.content.parts[0].text = new_text

    if callback_context is not None:
        callback_context.state["search_results_executive"] = new_text

    return llm_response


# ----------------------------
#  SQLite persistence for users/articles
# ----------------------------

DB_PATH = Path(__file__).parent / "ai_newsletter.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            profile_json TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS article_index (
            user_email TEXT,
            article_id TEXT,
            url TEXT,
            published_at TEXT,
            title TEXT,
            summary TEXT,
            PRIMARY KEY (user_email, article_id)
        );
        """
    )

    conn.commit()
    conn.close()
    print("✅ SQLite initialized at:", DB_PATH)

init_db()


def get_db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


# ---------- User profiles ----------


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


# ---------- Article index ----------


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

def writer_before_agent_callback(callback_context: CallbackContext):
    """
    Called before NewsletterWriter runs.
    - Increments loop iteration
    - Stashes start_time for latency measurement
    """
    state = callback_context.state

    # Track loop iteration (used by LoopAgent logic / verifier)
    state["loop_iteration"] = int(state.get("loop_iteration", 0)) + 1

    metrics = state.setdefault("metrics", {})
    writer_metrics = metrics.setdefault("NewsletterWriter", {})

    writer_metrics["iteration"] = state["loop_iteration"]
    writer_metrics["start_time"] = time.time()


def writer_after_agent_callback(callback_context: CallbackContext):
    """
    Called after NewsletterWriter runs.
    - Computes latency
    - Optionally captures token usage (if available)
    - Then delegates to save_state_after_agent_callback to persist state
    """
    from .utility import save_state_after_agent_callback  # if not in same file, adjust import

    state = callback_context.state
    metrics = state.setdefault("metrics", {})
    writer_metrics = metrics.setdefault("NewsletterWriter", {})

    end_time = time.time()
    start_time = writer_metrics.get("start_time")
    if start_time is not None:
        writer_metrics["latency_sec"] = end_time - start_time

    # Optional: try to grab token usage from the response if ADK exposes it
    resp = getattr(callback_context, "response", None)
    usage = getattr(resp, "usage_metadata", None) if resp is not None else None
    if usage is not None:
        # adjust these attribute names if your ADK version differs
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)

        if prompt_tokens is not None:
            writer_metrics["prompt_tokens"] = prompt_tokens
        if output_tokens is not None:
            writer_metrics["output_tokens"] = output_tokens
            writer_metrics["total_tokens"] = (prompt_tokens or 0) + output_tokens

    # Finally, do what you were already doing: save state
    save_state_after_agent_callback(callback_context)
