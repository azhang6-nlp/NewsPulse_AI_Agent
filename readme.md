<img width="100" height="106" alt="image" src="https://github.com/user-attachments/assets/5d543524-726e-411c-aa37-c3b43c46aec1" />
 
<h1 style="font-size:36px;">NewsPulse AI Agent</h1>

<h2 style="font-size:28px;">Multi-Agent Personalized AI/ML Newsletter System (Powered by Google ADK)</h2>



NewsPulse AI is an autonomous, self-correcting multi-agent research and newsletter system.
It profiles the user, retrieves fresh news, summarizes and verifies content, generates a beautiful HTML newsletter, evaluates quality with an LLM loop, learns from feedback, and continuously adapts over time.

Built using Google ADK’s LLM-powered multi-agent orchestration framework.

<h3 style="font-size:22px;">📘 1. Project Overview</h3>

**Project Name**: NewsPulse AI Agent 

**Platform**: Google ADK (LLM-powered multi-agent orchestrator)

**Team**:

Shuai Tan — Product Manager

Andy Zhang — Developer

Vivien Li — Developer

Adelie Yang — Developer

<h3 style="font-size:22px;">⚠️ 2. Problem Statement</h3>

Executives and AI professionals are overwhelmed with information noise. They receive hundreds of generic alerts each day, yet:

* Can't filter signal from noise

* Lack personalized, domain-aware summaries

* Face hallucination risks with most generative AI tools

* Lose productivity searching for relevant insights

They need a personalized, accurate, and timely intelligence brief that is always verified.

<h3 style="font-size:22px;">🎯 3. Product Vision & Solution</h3>

**Goal:** 
  Deliver the most relevant, timely, and trustworthy AI/ML news — with zero hallucinations.

**Solution:** 
  NewsPulse is a multi-agent analyst with:

* Sequential + loop agent patterns

* Self-evaluation / self-correction

* Personalized planning

* Memory-enhanced learning

* Fact verification

* Email delivery with feedback adaptation

**Value Proposition**

✔ Personalized content
✔ Fully citation-verified
✔ AI-generated HTML newsletter
✔ Learns from feedback
✔ Low latency end-to-end

<h3 style="font-size:22px;"> 📈 4. Success Metrics</h3>

**KPI	Target**

* Relevance	≥80% match to user interests
* Accuracy & Safety	100% citation-backed news
* Latency	< 4 minutes total
* Satisfaction	≥80% positive feedback

<h3 style="font-size:22px;"> 👤 5. User Stories</h3>

**Marketing Director**: Wants daily competitor insights aligned with strategic focus.

**CFO**: Wants clear, factual regulatory & financial summaries.

**Developer**: Wants a personalized newsletter remembering their own keywords.

<h3 style="font-size:22px;"> 🧠 6. Multi-Agent Architecture (Unified System)</h3> 

The system uses sequential and loop agent patterns.

Below merges both the PRD pipeline + your AI_Newsletter workflow.


#### 1). **UserProfilerAgent**

**Input:** user self-description + email + request

**Function:** LLM interprets interests, technical level, preferred tone (casual vs technical), preferred length, detailed request for a newsletter
  
**Output:** user profile JSON

---

#### 2). **HistoricalRecomenderAgent**

Slightly modify the user's requests to prevent stagnation and introduce related, novel concepts for current search.

**Output:** modified user profile JSON

---

#### 3). **PlannerAgent**

**Input:** user profile + today’s date

**Decides:**

* which topics to search
* which sources to pull from
* max number of articles
* newsletter outline

---

#### 4). **GoogleSearchAgent**

**Function:**
Fetch new articles 

**Output:** 
title, url, summary, published_time, uuid

---

#### 5). **FetchAgent**

* Fetch web page based on url from search agent

---

#### 6). **SummarizationAgent**

For each article:

* Extract key points
* Condense technical content into the correct level for this user


---

#### 7). **WriterAgent**

**Input:** user profile + summarized content
**Output:**
A fully drafted HTML newsletter:

* Title
* Section headers
* Summaries
* Tone adapted to user preferences

---

#### 8). **VerificationAgent** 

Checks the draft by comparing the summary against the related full text web page for:

* Tone correctness
* Accuracy of summaries
* If inaccurate, modified version

If bad → returns feedback + rewrite prompt → **WriterAgent rewrites → Evaluator re-checks**
This forms a **loop agent** pattern.

---

#### 9). **DeliveryAgent**

Uses an email-sending API (OpenAPI tool) to send HTML newsletters.
Logs:

* send time
* message_id
* delivery status

---

#### 10). **FeedbackAgent**

Parses user replies or click tracking:

* updates user profile preferences
* updates interest weights

This forms a **continuous improvement loop** driven by user behavior.

---

<h3 style="font-size:22px;"> 🏗️ 7. Architecture Diagram</h3>
<img width="613" height="559" alt="image" src="https://github.com/user-attachments/assets/49bd7211-9fe5-4e98-91ac-906e8b82ada8" />




**ASCII Diagram** <br>
<img width="532" height="240" alt="image" src="https://github.com/user-attachments/assets/3b3a397a-7108-4a73-8e9f-5f7a9209c853" />


<h3 style="font-size:22px;">📁 8. Project Folder Structure </h3> 

```
AI_Newsletter_Project/
├── .DS_Store
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── readme.md
├── readme_deployment_gcp.md
├── readme_setup_email_service.md
└── AI_Newsletter/
    ├── __init__.py
    ├── agent.py
    ├── logger_config.py
    ├── newsletter_agents.py
    ├── prompt.py
    ├── schema.py
    ├── utility.py
    └── vectors.py
```

<h3 style="font-size:22px;">⚙️ 9. Installation & Setup </h3> 
➡️ Using uv (recommended) <br>

* Create virtual env
uv venv --python 3.12 --seed
source .venv/bin/activate

* Install project
uv pip install .

<h3 style="font-size:22px;">🌐 10. Test in Google ADK Dev UI </h3>  

* Add your API key to .env.

* Run: uv run adk web

* Open browser: http://localhost:8000

  Select:

✔ AI_Newsletter (root agent)
✔ Run full agent workflow interactively

<h3 style="font-size:22px;"> 🔥 11. Test With FastAPI UI </h3> 

* Activate env: source .venv/bin/activate

* Install (if not done): uv pip install .

* Run FastAPI: uvicorn AI_Newsletter.main:app --reload

* Open: ➡️ http://127.0.0.1:8000


<h3 style="font-size:22px;">  ✉️ 12. Deployment & Email Services </h3>

Please find:

* readme_setup_email_service.md

* readme_deployment_gcp.md

* Supports deployment to GCP Cloud Run.

<h3 style="font-size:22px;"> 🧪 13. Sample Verification Agent Output </h3>

{
  "VerificationOutput": [
    {
      "sentence": "The latest AI advancements are democratizing coding...",
      "uuid": "short_blurb",
      "accuracy_or_not": true,
      "modified_version": "The latest AI advancements...",
      "justification": "Statement accurately reflects referenced content..."
    }
  ]
}

<h3 style="font-size:22px;">  🎓 14. How This Matches Multi-Agent System Concepts </h3>

✔ Sequential agents <br>
✔ Loop agent pattern (Writer ↔ Verifier)<br>
✔ Tools (Search, Crawler, Email Sender, Feedback Parser)<br>
✔ Long-term memory (SQLite + ChromaDB)<br>
✔ LLM Judge (VerificationAgent)<br>
✔ Structured A2A messaging <br>

📦 One-line Summary for Coursework

A fully personalized AI/ML newsletter generator using sequential, parallel, and loop agents, retrieval tools, long-term memory, structured A2A messaging, and LLM-based evaluation to generate, verify, and deliver daily intelligence briefs.
