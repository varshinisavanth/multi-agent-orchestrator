"""
The orchestrator: a hand-rolled state machine coordinating three agents.

    Planner  -> breaks the task into steps
    Coder    -> implements the plan
    Critic   -> reviews; if rejected, feedback goes back to Coder
                (bounded by max_revisions, so it can't loop forever)

Every handoff between agents is recorded in a TraceEvent list, so the
full multi-agent "conversation" is inspectable after the run -- this is
what you show an interviewer, not just the final code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from groq import Groq

from .agents import CoderAgent, CriticAgent, PlannerAgent, Plan, CodeSubmission, Review


@dataclass
class TraceEvent:
    agent: str
    action: str
    detail: str


@dataclass
class OrchestrationResult:
    success: bool
    final_code: str | None
    revisions_used: int
    trace: list[TraceEvent] = field(default_factory=list)


class Orchestrator:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", max_revisions: int = 3):
        client = Groq(api_key=api_key)
        self.planner = PlannerAgent(client, model)
        self.coder = CoderAgent(client, model)
        self.critic = CriticAgent(client, model)
        self.max_revisions = max_revisions

    def run(self, task: str) -> OrchestrationResult:
        trace: list[TraceEvent] = []

        plan: Plan = self.planner.plan(task)
        trace.append(
            TraceEvent("Planner", "created_plan", f"{len(plan.steps)} steps: {plan.reasoning}")
        )

        submission: CodeSubmission | None = None
        review: Review | None = None
        prior_code: str | None = None
        feedback: str | None = None

        for attempt in range(1, self.max_revisions + 1):
            submission = self.coder.implement(task, plan, prior_code, feedback)
            trace.append(
                TraceEvent(
                    "Coder", f"submitted_v{attempt}", submission.explanation
                )
            )

            review = self.critic.review(task, submission)
            trace.append(
                TraceEvent(
                    "Critic",
                    "approved" if review.approved else "rejected",
                    review.feedback,
                )
            )

            if review.approved:
                return OrchestrationResult(
                    success=True,
                    final_code=submission.code,
                    revisions_used=attempt,
                    trace=trace,
                )

            prior_code = submission.code
            feedback = review.feedback

        # exhausted revisions without approval -- return the last attempt,
        # clearly marked as not approved, rather than silently pretending success
        trace.append(
            TraceEvent(
                "Orchestrator",
                "gave_up",
                f"Exhausted {self.max_revisions} revisions without critic approval.",
            )
        )
        return OrchestrationResult(
            success=False,
            final_code=submission.code if submission else None,
            revisions_used=self.max_revisions,
            trace=trace,
        )


def format_trace(trace: list[TraceEvent]) -> str:
    lines = []
    for event in trace:
        lines.append(f"[{event.agent}] {event.action}: {event.detail}")
    return "\n".join(lines)
