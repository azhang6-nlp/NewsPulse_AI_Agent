from google.adk.agents import SequentialAgent, ParallelAgent

from .newsletter_agents import (requirement_agent,
                                planner_agent,
                                executive_search_agent,
                                executive_fetch_agent,
                                executive_summary_agent,
                                business_search_agent,
                                business_fetch_agent,
                                business_summary_agent,
                                NewsletterWriter)


planning_pipeline = SequentialAgent(
    name="planning_pipeline",
    sub_agents=[
                requirement_agent,
                planner_agent,
                ],
)

executive_summary_pipeline_agent = SequentialAgent(
    name="executive_summary_pipeline",
    sub_agents=[
        executive_search_agent,
        executive_fetch_agent, 
        executive_summary_agent
        ],
)
business_summary_pipeline_agent = SequentialAgent(
    name="business_summary_pipeline",
    sub_agents=[
                business_search_agent,
                business_fetch_agent, 
                business_summary_agent
                ],
)

parallel_summary_pipeline= ParallelAgent(
    name="parallel_summary_pipeline",
    sub_agents=[executive_summary_pipeline_agent, business_summary_pipeline_agent],
)

newsletter_writing_verifcation_pipeline_agent = SequentialAgent(
    name="newsletter_writing_verifcation_pipeline",
    sub_agents=[
                NewsletterWriter,
                ],
)

root_agent = SequentialAgent(
    name="AI_Newsletter_Agent",
    sub_agents=[
                planning_pipeline,
                parallel_summary_pipeline,
                newsletter_writing_verifcation_pipeline_agent 
                ],
)
