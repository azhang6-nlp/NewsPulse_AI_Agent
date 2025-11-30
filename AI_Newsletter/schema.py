from typing import List, Literal, Optional
from pydantic import BaseModel, AnyUrl, Field

class NewsletterProfileOutput(BaseModel):
    email: str = Field(
        ...,
        description="User's email address inferred from the message."
    )
    technical_level: Literal["beginner", "intermediate", "expert"] = Field(
        ...,
        description="User's technical proficiency inferred from the message."
    )
    preferred_sources: List[str] = Field(
        default_factory=list,
        description="List of preferred information-source domains. Empty list if none provided."
    )
    detailed_request: str = Field(
        ...,
        description="A natural-language description summarizing the user's detailed requirements for the newsletter."
    )


class TaskDelegationPlan(BaseModel):
    executive_summary_agent: str

class subsection(BaseModel):
    subsection: str

class SectionOutline(BaseModel):
    executive_summary: List[subsection]
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
    search_queries: List[search_query_item]
    task_delegation_plan: TaskDelegationPlan
    section_outline: SectionOutline

class SummaryItem(BaseModel):
    topic: str = Field(..., description="The topic associated with this page")
    title: str = Field(..., description="The title associated with this page")
    final_url: str = Field(..., description="The URL of the page summarized")
    uuid: str = Field(..., description="The unique identifier for the page, passed on from search and fetch agent")
    publish_date: str = Field(..., description="publish date of the webpage")
    summary: str = Field(..., description="A concise summary of the page's full text")

class SummaryOutput(BaseModel):
    summaries: List[SummaryItem] = Field(
        ..., description="List of summarized items for each topic / URL"
    )
    summary_done: bool = False

class SectionItem(BaseModel):
    heading: str
    body: str
    final_url: str
    uuid: str


class NewsletterOutput(BaseModel):
    newsletter_title: str = Field(..., description="Short headline for the newsletter")
    date: str = Field(..., description="Publication date (YYYY-MM-DD), get this newsletter date from session state current_date field or infer from detailed_request in the profile")
    short_blurb: str = Field(..., description="One-sentence summary / lede")
    executive_summary: List[SectionItem] = Field(..., description="Key executive summaries (heading & body)")
    business_implications: List[SectionItem] = Field(..., description="Key business/industry implications (heading & body)")
    citations: List[str] = Field(..., description="List of source URLs / short citations, got from url in the inputs of executive summary and business summary")
    tl_dr: str = Field(..., description="TL;DR (3 bullet lines)")
    call_to_action: str = Field(..., description="One recommended next step for readers")

class AccuracyCheckItem(BaseModel):
    sentence: str = Field(
        ...,
        description="The original sentence to evaluate."
    )
    uuid: str
    accuracy_or_not: bool = Field(
        ...,
        description="Whether the sentence is accurate (true) or inaccurate (false)."
    )
    modified_version: str = Field(
        ...,
        description="A corrected or improved version of the sentence."
    )
    justification: str = Field(
        ...,
        description="Explanation detailing why the sentence is accurate or inaccurate."
    )

class VerificationItem(BaseModel):
    sentence: str
    uuid: str
    accuracy_or_not: bool
    modified_version: Optional[str] = ""  # 👈 allow None, default empty string
    justification: str

class VerificationOutput(BaseModel):
    VerificationOutput: List[AccuracyCheckItem]

