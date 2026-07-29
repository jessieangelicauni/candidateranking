from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""


class WorkHistoryEntry(BaseModel):
    title: str
    company: str
    start_date: str
    end_date: str
    achievements: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    degree: str
    institution: str
    year: str


class ProjectEntry(BaseModel):
    name: str
    description: str
    tech: list[str] = Field(default_factory=list)


class JDRequirements(BaseModel):
    title: str
    required_skills: list[str] = Field(
        description=(
            "Every mandatory skill, technology, or technique the JD names - from bulleted "
            "lists AND from prose (e.g. 'developing backend applications' names Backend "
            "Development as a skill just as much as a bullet would). Exhaustive, not a sample."
        )
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Every preferred/bonus skill, technology, or technique the JD names. Skills only - "
            "never degrees or generic experience statements, those belong in education instead."
        ),
    )
    min_experience_years: int = 0
    education: str = Field(
        default="",
        description=(
            "The degree/field-of-study requirement as one string, copied close to as stated "
            "(e.g. \"Bachelor's or Master's degree in Computer Science, Data Science, "
            "Engineering, or a related field\"). Must be non-empty whenever the JD mentions a "
            "degree anywhere, even under a heading like 'Preferred Qualifications' that isn't "
            "literally labeled 'Education'. Never split into multiple degree/field combinations."
        ),
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description=(
            "Every distinct duty the JD describes, one per list item - including duties stated "
            "in prose without a 'Responsibilities' heading."
        ),
    )


class ExtractedProfileFields(BaseModel):
    contact: ContactInfo
    skills: list[str]
    work_history: list[WorkHistoryEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)


class CandidateProfile(ExtractedProfileFields):
    candidate_id: str
    raw_cv_text: str


class PrefilterResult(BaseModel):
    candidate_id: str
    similarity: float
    passed: bool


class Tier(str, Enum):
    STRONG_FIT = "Strong Fit"
    MODERATE_FIT = "Moderate Fit"
    WEAK_FIT = "Weak Fit"
    NOT_A_FIT = "Not a Fit"


class EvidenceClaim(BaseModel):
    claim: str
    quote: str


class JudgeVerdict(BaseModel):
    tier: Tier
    rating: int = Field(ge=1, le=10)
    evidence: list[EvidenceClaim]

    @field_validator("evidence")
    @classmethod
    def _drop_blank_quotes(cls, evidence: list[EvidenceClaim]) -> list[EvidenceClaim]:
        return [claim for claim in evidence if claim.quote.strip()]


class JudgeResult(JudgeVerdict):
    candidate_id: str


class CalibratedResult(BaseModel):
    candidate_id: str
    final_rank: int
    tier: Tier
    rating: int
    calibration_notes: str


class CalibrationOutput(BaseModel):
    results: list[CalibratedResult]


class HallucinationReport(BaseModel):
    candidate_id: str
    unverified_quotes: list[str] = Field(default_factory=list)

    @property
    def all_verified(self) -> bool:
        return len(self.unverified_quotes) == 0
