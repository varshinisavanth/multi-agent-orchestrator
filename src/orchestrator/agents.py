"""
Three agent roles, each a focused wrapper around the Claude API with its
own system prompt and structured JSON output. Kept intentionally simple
(no framework) so the handoff logic in orchestrator.py stays fully
visible and explainable in an interview.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from groq import Groq


@dataclass
class Plan:
    steps: list[str]
    reasoning: str


@dataclass
class CodeSubmission:
    code: str
    explanation: str


@dataclass
class Review:
    approved: bool
    feedback: str


class PlannerAgent:
    """Breaks a task description down into a small ordered list of
    concrete implementation steps."""

    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model = model

    def plan(self, task: str) -> Plan:
        system = (
            "You are a planning agent. Break the given coding task into "
            "3-6 concrete, ordered implementation steps. Respond with ONLY "
            "a JSON object, no markdown fences: "
            '{"steps": ["step 1", "step 2", ...], "reasoning": "<1-2 sentences>"}'
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Task: {task}"},
            ],
        )
        parsed = json.loads(_text(response))
        return Plan(steps=parsed["steps"], reasoning=parsed["reasoning"])


class CoderAgent:
    """Writes code implementing the plan, and revises based on critic
    feedback on subsequent iterations."""

    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model = model

    def implement(
        self, task: str, plan: Plan, prior_code: str | None = None, feedback: str | None = None
    ) -> CodeSubmission:
        system = (
            "You are a coding agent. Implement the given task following the "
            "provided plan. Respond with ONLY a JSON object, no markdown "
            'fences: {"code": "<complete code as a string>", '
            '"explanation": "<1-2 sentences on your implementation>"}'
        )
        user = f"Task: {task}\n\nPlan steps:\n" + "\n".join(f"- {s}" for s in plan.steps)
        if prior_code and feedback:
            user += (
                f"\n\nYour previous submission:\n```\n{prior_code}\n```\n"
                f"Critic feedback to address:\n{feedback}"
            )

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        parsed = json.loads(_text(response))
        return CodeSubmission(code=parsed["code"], explanation=parsed["explanation"])


class CriticAgent:
    """Reviews a code submission against the original task and either
    approves it or gives specific, actionable feedback for revision."""

    def __init__(self, client: Groq, model: str):
        self.client = client
        self.model = model

    def review(self, task: str, submission: CodeSubmission) -> Review:
        system = (
            "You are a strict code review agent. Review the submitted code "
            "against the task. Check correctness, edge cases, and clarity. "
            "Only approve if the code genuinely satisfies the task. Respond "
            'with ONLY a JSON object, no markdown fences: {"approved": '
            '<true/false>, "feedback": "<specific, actionable feedback, or '
            'a short approval note if approved>"}'
        )
        user = f"Task: {task}\n\nSubmitted code:\n```\n{submission.code}\n```"

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        parsed = json.loads(_text(response))
        return Review(approved=bool(parsed["approved"]), feedback=parsed["feedback"])


def _text(response) -> str:
    return response.choices[0].message.content
