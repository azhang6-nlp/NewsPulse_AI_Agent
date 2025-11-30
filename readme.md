
# 📰 AI Newsletter – Multi-Agent Personalized News System

A multi-agent system that profiles users, retrieves fresh AI/ML news, summarizes content, generates a personalized HTML newsletter, evaluates quality with an LLM loop, and finally delivers the email — with feedback incorporated back into long-term memory.


AI_Newsletter/
  ├── __init__.py
  ├── agent.py              # <- ADK entrypoint (defines root_agent)
  ├── newsletter_agents.py  # define all the adk agents used by the root_agent (see below for more info)
  └── utility.py            # user profile storage; real email integration; funtion tools;  other functions for state management
  └── schema.py             # define output schemas for various agents
  └── logger_config.py      # logging for sessions

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

# Refer to readme_setup_email_service.md for email sending service
---

# Refer to readme_deployment_gcp.md for deploy the agent to GCP Cloud run

---

# 🧠 Multi-Agent Flow (Core System Architecture)

This system uses a **multi-agent pipeline** with sequential, parallel, and loop patterns.

---

## 1. **UserProfilerAgent**

**Input:** user self-description + email + request
**Function:**

* LLM interprets interests, technical level, preferred tone (casual vs technical), preferred length, detailed request for a newsletter
  **Output:** user profile JSON

---

## 2. **HistoricalRecomenderAgent**

Slightly modify the user's requests to prevent stagnation and introduce related, novel concepts for current search.
**Output:** modified user profile JSON

---

## 3. **PlannerAgent**

**Input:** user profile + today’s date
**Decides:**

* which topics to search
* which sources to pull from
* max number of articles
* newsletter outline

---

## 4. **GoogleSearchAgent**

Uses tools such as:

* Google Search Tool

**Function:**
Fetch new articles 

**Output:** title, url, summary, published_time, uuid

---

## 5. **FetchAgent**

* Fetch web page based on url from search agent

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

## 8. **VerificationAgent** 

Checks the draft by comparing the summary against the related full text web page for:

* Tone correctness
* Accuracy of summaries
* If inaccurate, modified version

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

* updates user profile preferences
* updates interest weights

This forms a **continuous improvement loop** driven by user behavior.

---

# 🎓 How This Matches Course Key Concepts

## ✔ Multi-Agent System Types

* **Sequential agents:** full pipeline from UserProfiler → Delivery
* **Loop agent:** Writer ↔ Evaluation self-correction cycle

## ✔ Tools

* google_search_tool
* crawler / RSS tool
* email_sender_tool
* feedback_parser_tool

## ✔ Memory

* Long-term: user profile
* Memory bank: stored in SQLite + ChromaDB

## ✔ Agent Evaluation

* Online: LLM judge (VerificationAgent)
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


# Sample Verification Output 
{"VerificationOutput": [{"sentence": "The latest AI advancements are democratizing coding through 'vibe coding' and enhancing developer experience with powerful models like Anthropic's Claude Opus 4.5, offering profound implications for healthcare insurance.", "uuid": "short_blurb", "accuracy_or_not": true, "modified_version": "The latest AI advancements are democratizing coding through 'vibe coding' and enhancing developer experience with powerful models like Anthropic's Claude Opus 4.5, offering profound implications for healthcare insurance.", "justification": "The statement accurately reflects the content from both provided references. The Times of India article discusses how 'vibe coding' democratizes tech for non-technical individuals, and the Anthropic article introduces Claude Opus 4.5 as an advanced AI model enhancing developer experience. The 'profound implications for healthcare insurance' is a valid synthesis of the information, especially considering the detailed business implications provided in the newsletter context, which were themselves verified."}]}

