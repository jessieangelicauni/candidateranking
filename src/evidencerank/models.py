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
    required_skills: list[str]
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_experience_years: int = 0
    education: str = ""
    responsibilities: list[str] = Field(default_factory=list)


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
