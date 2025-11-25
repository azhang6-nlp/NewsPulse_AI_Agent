"""
AI_Newsletter.agent

Multi-agent AI newsletter pipeline using Google ADK.

Run with ADK dev UI:
    cd AI_Newsletter
    adk web AI_Newsletter

Then pick "AI_Newsletter" (root_agent) in the UI.
"""
from __future__ import annotations

from typing import List, Dict, Any
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime # Imported for use in send_newsletter_email

from google.adk.agents import (
    BaseAgent,
    LlmAgent,
    SequentialAgent,
    ParallelAgent,
    LoopAgent,
)
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.function_tool import FunctionTool
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from AI_Newsletter.storage import (
    load_user_profile,
    save_user_profile,
    get_seen_article_ids,
    save_article_for_user,
)
from AI_Newsletter.vectors import add_articles, search_similar

# -------------------------------------------------------------------
# 📌 Shared State Keys
# -------------------------------------------------------------------

MAX_ARTICLES             = 5
STATE_USER_PROFILE       = "user_profile"          # dict
STATE_REFINED_TOPICS     = "refined_topics"        # list[str] <--- NEW KEY
STATE_DAILY_PLAN         = "daily_plan"            # dict
STATE_INDEXED_ARTICLES   = "indexed_articles"      # list[dict]
STATE_SUMMARIES          = "article_summaries"     # list[dict]
STATE_NEWSLETTER_HTML    = "newsletter_html"       # str
STATE_EVAL_NOTES         = "newsletter_eval_notes" # str

# -------------------------------------------------------------------
# 🛠  Core Python tools (SQLite + Chroma + email + feedback)
# -------------------------------------------------------------------

def index_articles_for_user(email: str, articles: List[ Dict[str, Any] ]) -> Dict[str, Any]:
    """
    Filter out already-seen articles, save new ones to SQLite, and optionally index
    into the vector DB if an 'embedding' field is present.
    """
    seen_ids = set(get_seen_article_ids(email))

    new_articles: List[Dict[str, Any]] = []
    for a in articles:
        aid = a.get("id") or a.get("url")
        if not aid or aid in seen_ids:
            continue
        a["id"] = aid
        save_article_for_user(email, a)
        new_articles.append(a)

    # If embeddings exist, push to vector DB
    articles_with_emb = [a for a in new_articles if "embedding" in a]
    if articles_with_emb:
        add_articles(articles_with_emb)

    return {
        "total_new": len(new_articles),
        "new_articles": new_articles,
    }


def semantic_search_articles(query_embedding: List[float], top_k: int = MAX_ARTICLES) -> Dict[str, Any]:
    """Thin wrapper around the vector DB semantic search."""
    return search_similar(query_embedding=query_embedding, top_k=top_k)


# The original imports for smtplib, etc., are at the top,
# removed redundant re-imports here for cleanliness.


def send_newsletter_email(to_email: str, subject: str, html: str) -> dict:
    """
    Send HTML email via Gmail SMTP OR run in demo mode.
    (Function remains unchanged from your original file)
    """
    demo_mode = os.getenv("NEWSLETTER_DEMO_MODE", "1").lower() in ("1", "true", "yes")

    # -----------------------
    # DEMO MODE (no real SMTP)
    # -----------------------
    if demo_mode:
        print("=== [DEMO] send_newsletter_email called ===")
        print("To:", to_email)
        print("Subject:", subject)

        # Optionally save the HTML so you can open it in a browser
        debug_dir = Path("debug_newsletters")
        debug_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = debug_dir / f"newsletter_{ts}.html"
        try:
            fname.write_text(html, encoding="utf-8")
            print(f"=== [DEMO] Saved newsletter HTML to {fname}")
        except Exception as e:
            print(f"=== [DEMO] Failed to write HTML file: {e}")

        return {
            "status": "mock_sent",
            "message_id": f"demo-{ts}",
            "debug_file": str(fname),
        }

    # -----------------------
    # REAL SMTP MODE
    # -----------------------
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_email = os.getenv("NEWSLETTER_FROM_EMAIL", user)

    if not all([host, port, user, password, from_email]):
        # Don't raise – return an error dict so the agent can surface it
        return {
            "status": "config_error",
            "message": "SMTP config missing; check .env (HOST/PORT/USER/PASS/FROM).",
        }

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    msg.attach(MIMEText("Your email client does not support HTML.", "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)  # IMPORTANT: must be an app password
            server.sendmail(from_email, [to_email], msg.as_string())

        return {"status": "sent", "message_id": "gmail-smtp"}
    except smtplib.SMTPAuthenticationError as e:
        # Bad credentials / wrong app password
        return {
            "status": "auth_error",
            "message": f"SMTP authentication failed: {e}",
        }
    except Exception as e:
        # Any other SMTP/network error
        return {
            "status": "error",
            "message": f"SMTP error: {e}",
        }


def parse_feedback(raw_email_text: str) -> Dict[str, Any]:
    """
    Very simple heuristic feedback parser.
    (Function remains unchanged from your original file)
    """
    text = (raw_email_text or "").lower()
    feedback = {
        "too_long": "too long" in text,
        "too_short": "too short" in text,
        "too_basic": "too basic" in text,
        "too_advanced": "too advanced" in text,
        "liked_topics": [],
        "disliked_topics": [],
    }
    return feedback

# -------------------------------------------------------------------
# 🧰 Wrap Python functions as ADK FunctionTools
# -------------------------------------------------------------------

load_user_profile_tool = FunctionTool(
    load_user_profile)

save_user_profile_tool = FunctionTool(
    save_user_profile)

index_articles_tool = FunctionTool(
    index_articles_for_user)

semantic_search_tool = FunctionTool(
    semantic_search_articles)

send_newsletter_tool = FunctionTool(
    send_newsletter_email)

parse_feedback_tool = FunctionTool(
    parse_feedback)

# -------------------------------------------------------------------
# 🧪 Optional: LangChain RAG tool (guarded import)
# -------------------------------------------------------------------

HAVE_LANGCHAIN = False
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableSequence

    HAVE_LANGCHAIN = True
except ImportError:
    HAVE_LANGCHAIN = False


def _build_langchain_rag_chain() -> RunnableSequence:
    # ... (LangChain implementation omitted for brevity, remains unchanged) ...
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        collection_name="articles",
        embedding_function=embeddings,
        persist_directory=str(Path(__file__).parent / "chroma_store"),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": MAX_ARTICLES})

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI/ML newsletter assistant answering a user's question
        using ONLY the provided context.

        Context:
        {context}

        Question:
        {question}

        Answer in a concise, newsletter-friendly style.
        """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    rag_chain: RunnableSequence = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
    )
    return rag_chain


def langchain_rag_answer(question: str) -> Dict[str, Any]:
    # ... (LangChain implementation omitted for brevity, remains unchanged) ...
    if not HAVE_LANGCHAIN:
        return {
            "answer": (
                "LangChain is not installed in this environment, "
                "so the RAG tool is unavailable."
            ),
            "source_urls": [],
        }

    rag_chain = _build_langchain_rag_chain()
    result = rag_chain.invoke({"question": question})
    answer_text = getattr(result, "content", str(result))

    # Collect URLs from retrieved docs for citations
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        collection_name="articles",
        embedding_function=embeddings,
        persist_directory=str(Path(__file__).parent / "chroma_store"),
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": MAX_ARTICLES})
    docs: List[Document] = retriever.get_relevant_documents(question)
    urls = [d.metadata.get("url") for d in docs if d.metadata.get("url")]

    return {
        "answer": answer_text,
        "source_urls": urls,
    }


langchain_rag_tool = FunctionTool(langchain_rag_answer)

# -------------------------------------------------------------------
# 🤖 1. UserProfilerAgent
# -------------------------------------------------------------------

user_profiler_agent = LlmAgent(
    name="UserProfilerAgent",
    model="gemini-2.5-flash",
    description="Profiles the user (interests, technical level, tone, length, and preferred sources).",
    # ... (Instruction and tools remain unchanged) ...
    instruction="""
You are a user profiling agent for an AI/ML newsletter.

The LAST user message contains:
- Their self-description
- Their email
- Optionally, a list of preferred or trusted sources (e.g. 'openai.com, arxiv.org, anthropic.com').

Task:
1. Read the message.
2. Infer:
   - email (string)
   - topics (list of 3–10 strings like "LLM evaluation", "MLOps", "RecSys")
   - technical_level in ["beginner", "intermediate", "expert"]
   - tone in ["casual", "technical", "mixed"]
   - length_preference in ["short", "medium", "long"]
   - preferred_sources: a list of domain strings like ["openai.com", "arxiv.org"].
     If the user did not provide any, use an empty list [].

3. Call the save_user_profile tool with this JSON to persist it.

4. Return ONLY valid JSON (no prose). Example:

{
  "email": "user@example.com",
  "topics": ["LLM evaluation", "MLOps"],
  "technical_level": "expert",
  "tone": "mixed",
  "length_preference": "medium",
  "preferred_sources": ["openai.com", "arxiv.org"]
}
""",
    tools=[save_user_profile_tool],
    output_key=STATE_USER_PROFILE,
)


# -------------------------------------------------------------------
# 🤖 2. HistoricalRecommenderAgent (NEW!)
# -------------------------------------------------------------------

historical_recommender_agent = LlmAgent(
    name="HistoricalRecommenderAgent",
    model="gemini-2.5-flash",
    description="Adjusts the user's topics to include novelty and drift based on related articles in the vector database.",
    tools=[semantic_search_tool, load_user_profile_tool], # load is optional but good for context
    instruction=f"""
You are the Historical Recommender. Your goal is to slightly modify the user's
topic list to prevent stagnation and introduce related, novel concepts for today's search.

Inputs from state:
- "{STATE_USER_PROFILE}" (current profile topics)

Task:
1. Examine the user's current topics from state["{STATE_USER_PROFILE}"]["topics"].
2. Suggest 1-2 new, related topics that would introduce novelty but remain relevant to the user's core interests.
   *Example: If core topic is 'MLOps', suggest 'AIOps' or 'Model Serving Infrastructure'.*
3. The final list MUST include ALL original topics plus the 1-2 suggested topics.
4. Output ONLY the refined list of topics as a JSON list.

Return ONLY a JSON list (list of strings):
["topic 1", "topic 2", "newly suggested topic 3"]
""",
    output_key=STATE_REFINED_TOPICS,
)


# -------------------------------------------------------------------
# 🤖 3. DailyPlannerAgent (Updated)
# -------------------------------------------------------------------

daily_planner_agent = LlmAgent(
    name="DailyPlannerAgent",
    model="gemini-2.5-flash",
    description="Plans which topics, preferred sources, and max_articles to use today.",
    instruction=f"""
You are a planner for a daily AI/ML newsletter.

You have access to:
- The base user profile in state["{STATE_USER_PROFILE}"]
- The refined topics list in state["{STATE_REFINED_TOPICS}"] (use this for topic selection!)

Task:
- Select which topics to focus on today. (Use the list from "{STATE_REFINED_TOPICS}").
- Decide which preferred_sources to use (may be all or a subset).
- Decide whether to also use broader Google exploration.
- Cap the total article count at MAX {MAX_ARTICLES} for this demo project.

Return ONLY valid JSON in this shape:

{{
  "topics": ["...", "..."],
  "preferred_sources": ["openai.com", "arxiv.org"],
  "use_google_explore": true,
  "max_articles": {MAX_ARTICLES}
}}

Guidelines:
- Start from the topics in state["{STATE_REFINED_TOPICS}"]. You can narrow down to 1–3 topics for today.
- preferred_sources should come from user_profile["preferred_sources"], possibly trimmed to at most 3 domains.
- max_articles should be between 3 and {MAX_ARTICLES}.
- If preferred_sources is non-empty, usually set use_google_explore = true to mix trusted + discovery.
""",
    tools=[],
    output_key=STATE