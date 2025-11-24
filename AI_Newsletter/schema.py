from typing import List
from pydantic import BaseModel, AnyUrl, Field

class TaskDelegationPlan(BaseModel):
    executive_summary_agent: str
    industry_implications_agent: str  # note: probably meant "implications"

class subsection(BaseModel):
    subsection: str

class SectionOutline(BaseModel):
    executive_summary: List[subsection]
    technical_highlitghts: List[subsection]
    business_implications: List[subsection]

class search_query_item(BaseModel):
    topic: str

class clarify_question(BaseModel):
    question: str
    answer: str

class clarifications_needed(BaseModel):
    clarifications_needed: List[clarify_question]
    request_clarification_done: bool = False
    detailed_request: str = Field(..., description="the summarized request with inclusion of user's answers to the clarification questions.")

class NewsletterSections(BaseModel):
    search_queries_for_executive: List[search_query_item]
    search_queries_for_Industry_Implications: List[search_query_item]
    task_delegation_plan: TaskDelegationPlan
    section_outline: SectionOutline

class SummaryItem(BaseModel):
    topic: str = Field(..., description="The topic associated with this page")
    title: str = Field(..., description="The title associated with this page")
    url: str = Field(..., description="The URL of the page summarized")
    uuid: str = Field(..., description="The unique identifier for the page, passed on from search and fetch agent")
    summary: str = Field(..., description="A concise summary of the page's full text")

class SummaryOutput(BaseModel):
    summaries: List[SummaryItem] = Field(
        ..., description="List of summarized items for each topic / URL"
    )
    summary_done: bool = False

class SectionItem(BaseModel):
    heading: str
    body: str
    url: str


class NewsletterOutput(BaseModel):
    newsletter_title: str = Field(..., description="Short headline for the newsletter")
    date: str = Field(..., description="Publication date (YYYY-MM-DD), got from session state newsletter_date field")
    short_blurb: str = Field(..., description="One-sentence summary / lede")
    executive_summary: List[SectionItem] = Field(..., description="Key executive summaries (heading & body)")
    technical_highlights: List[SectionItem] = Field(..., description="Key technical highlights (heading & body)")
    business_implications: List[SectionItem] = Field(..., description="Key business/industry implications (heading & body)")
    citations: List[str] = Field(..., description="List of source URLs / short citations, got from url in the inputs of executive summary and business summary")
    tl_dr: str = Field(..., description="TL;DR (3 bullet lines)")
    call_to_action: str = Field(..., description="One recommended next step for readers")
