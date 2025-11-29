from google.adk.agents import SequentialAgent, ParallelAgent

import logging

# Import the setup function
from .logger_config import setup_file_logging
setup_file_logging(logging.DEBUG) # Call it first thing to start logging

from .newsletter_agents import (
    profile_agent,
    planner_agent,
    historical_recommender_agent, # Added
    feedback_agent,               # Added
    executive_search_agent,
    executive_fetch_agent,
    executive_summary_agent,
    NewsletterWriter,
    verification_agent,
    newsletter_dispatcher
)

# --- 1. Planning Pipeline (Updated) ---
# Goal: Load profile, refine topics, and create the plan.
planning_pipeline = SequentialAgent(
    name="planning_pipeline",
    sub_agents=[
        # 1. Profile Agent (Now loads/updates the profile)
        profile_agent,
        # 2. Refine topics based on history and novelty
        historical_recommender_agent,
        # 3. Create the daily plan
        planner_agent
    ],
)

# --- 2. Summary Pipeline (Content Ingestion) ---
# Goal: Fetch, summarize, and index articles.
summary_pipeline_agent = SequentialAgent(
    name="summary_pipeline",
    sub_agents=[
        executive_search_agent,
        executive_fetch_agent, 
        executive_summary_agent
    ],
)


# --- 3. Writing and Verification Pipeline ---
# Goal: Draft and quality-check the newsletter HTML.
newsletter_writing_verifcation_pipeline_agent = SequentialAgent(
    name="newsletter_writing_verifcation_pipeline",
    sub_agents=[
        NewsletterWriter,
        verification_agent # Assumed to handle the refinement loop/checker logic
    ],
)

# --- 4. Root Agent (Full Pipeline) ---
# Goal: Orchestrate the entire flow, ending with the feedback loop.
root_agent = SequentialAgent(
    name="AI_Newsletter_Agent",
    sub_agents=[
        # PHASE 1: Planning and Topic Refinement
        planning_pipeline,
        
        # PHASE 2: Content Generation
        summary_pipeline_agent,
        
        # PHASE 3: Writing and verification (Verification/Refinement)
        newsletter_writing_verifcation_pipeline_agent,

        # PHASE 4: Delivery
        newsletter_dispatcher,
        
        # PHASE 5: Feedback Loop (Runs at the end to process user reply)
        feedback_agent, 
    ],
)