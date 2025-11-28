from google.adk.agents import SequentialAgent, ParallelAgent

from .newsletter_agents import (profile_agent,
                                planner_agent,
                                executive_search_agent,
                                executive_fetch_agent,
                                executive_summary_agent,
                                NewsletterWriter,
                                verification_agent)


planning_pipeline = SequentialAgent(
    name="planning_pipeline",
    sub_agents=[
                profile_agent,
                planner_agent,
                ],
)

summary_pipeline_agent = SequentialAgent(
    name="summary_pipeline",
    sub_agents=[
        executive_search_agent,
        executive_fetch_agent, 
        executive_summary_agent
        ],
)


newsletter_writing_verifcation_pipeline_agent = SequentialAgent(
    name="newsletter_writing_verifcation_pipeline",
    sub_agents=[
                NewsletterWriter,
                verification_agent
                ],
)

root_agent = SequentialAgent(
    name="AI_Newsletter_Agent",
    sub_agents=[
                planning_pipeline,
                summary_pipeline_agent,
                newsletter_writing_verifcation_pipeline_agent 
                ],
)
