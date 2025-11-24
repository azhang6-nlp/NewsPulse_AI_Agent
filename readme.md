
# 📰 AI Newsletter – Multi-Agent Personalized News System

A multi-agent system that profiles users, retrieves fresh AI/ML news, summarizes content, generates a personalized HTML newsletter, evaluates quality with an LLM loop, and finally delivers the email — with feedback incorporated back into long-term memory.


AI_Newsletter/
  ├── __init__.py
  ├── agent.py          # <- ADK entrypoint (defines root_agent)
  ├── storage.py        # sqlite + chroma helpers
  └── email_sender.py   # real email integration

---

# 🚀 Installation (pyproject.toml)

Create and activate your env:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Install all project dependencies:

```bash
uv pip install .
```

---

# 🌐 Test on ADK UI

1. Put your own `api_key` into `.env`

2. Start the ADK dev UI:

   ```bash
   # adk web --reload
   uv run adk web
   ```

3. Open:
   **[http://localhost:8000/](http://localhost:8000/)**

4. Select the **AI_Newsletter** agent

5. Interact with the agent workflow from the browser UI

---

# 🔥 Test with FastAPI UI

Activate env:

```bash
source .venv/bin/activate
```

Install local project (if not done):

```bash
uv pip install .
```

Run FastAPI:

```bash
uvicorn AI_Newsletter.main:app --reload
```

Open:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

This UI uses:

* FastAPI
* Jinja2 templates
* SQLite for structured data
* ChromaDB for vector search

---

# 🧠 Multi-Agent Flow (Core System Architecture)

This system uses a **multi-agent pipeline** with sequential, parallel, and loop patterns.

---

## 1. **UserProfilerAgent**

**Input:** user self-description + email
**Function:**

* LLM interprets interests, technical level, preferred tone (casual vs technical), preferred length
  **Output:** user profile JSON

---

## 2. **ProfileStorageAgent**

Stores long-term user profile + maintains:

* `seen_article_ids` (avoid duplicates)
* basic stats like open rate / click rate

(Can be implemented as a tool or direct DB call.)

---

## 3. **DailyPlannerAgent**

Runs daily (cron / scheduler).

**Input:** user profile + today’s date
**Decides:**

* which topics to search
* which sources to pull from
* max number of articles

**Output example:**

```json
{
  "topics": ["LLM evaluation", "MLOps"],
  "sources": ["google_news", "specific_blog"],
  "max_articles": 5
}
```

---

## 4. **RetrieverAgent** (parallelizable)

Uses tools such as:

* Google Search Tool
* Custom website crawler / RSS

**Function:**
Fetch new articles → filter out anything already in `seen_article_ids`

**Output:** title, url, snippet, published_time

---

## 5. **IndexingAgent**

* Saves new article vectors to ChromaDB
* Updates `seen_article_ids`
* Creates long-term searchable memory index

(Exposed as a `save_article_index_tool`.)

---

## 6. **SummarizationAgent**

For each article:

* Extract key points
* Condense technical content into the correct level for this user

---

## 7. **WriterAgent**

**Input:** user profile + summarized content
**Output:**
A fully drafted HTML newsletter:

* Title
* Section headers
* Summaries
* Tone adapted to user preferences

---

## 8. **EvaluationAgent** (LLM self-correction loop)

Checks the draft for:

* Tone correctness
* Length too long / too short
* Quality of summaries
* Readability

If bad → returns feedback + rewrite prompt → **WriterAgent rewrites → Evaluator re-checks**
This forms a **loop agent** pattern.

---

## 9. **DeliveryAgent**

Uses an email-sending API (OpenAPI tool) to send HTML newsletters.
Logs:

* send time
* message_id
* delivery status

---

## 10. **FeedbackAgent**

Parses user replies or click tracking:

* detects “too long”, “too basic”, “love this topic”, etc.
* updates user profile preferences
* updates interest weights

This forms a **continuous improvement loop** driven by user behavior.

---

# 🎓 How This Matches Course Key Concepts

## ✔ Multi-Agent System Types

* **Sequential agents:** full pipeline from UserProfiler → Delivery
* **Parallel agents:** RetrieverAgent hitting multiple sources concurrently
* **Loop agent:** Writer ↔ Evaluation self-correction cycle

## ✔ Tools

* google_search_tool
* crawler / RSS tool
* article_index_tool
* email_sender_tool
* feedback_parser_tool

## ✔ Memory

* Long-term: user profile + seen_article_ids + click history
* Context engineering: summarization of long histories
* Memory bank: stored in SQLite + ChromaDB

## ✔ Long-running Operations

* DailyPlannerAgent invoked via cron / workflow engine
* Pause/resume when an API rate-limits or a crawler times out

## ✔ Agent Evaluation

* Online: LLM judge (EvaluationAgent)
* Offline: open-rate & click-through metrics

## ✔ Agent-to-Agent Protocol (A2A)

Structured JSON messages passed between agents:

```json
{
  "type": "DailyPlanResult",
  "topics": ["AI eval"],
  "max_articles": 5,
  "exclude_ids": ["123", "456"]
}
```

---

# 📦 One-Line Summary for Assignment Submission

> This project is a multi-agent personalized newsletter system using sequential, parallel, and loop agents, retrieval tools, long-term memory (SQLite + Chroma), daily planning, LLM-based evaluation, and structured A2A messaging to deliver customized AI/ML newsletters and adapt continuously from user feedback.

