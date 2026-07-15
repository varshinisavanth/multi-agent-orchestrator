from unittest.mock import MagicMock

from orchestrator.orchestrator import Orchestrator
from orchestrator.agents import Plan, CodeSubmission, Review


def build_orchestrator_with_mocks(max_revisions: int = 3) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)  # bypass __init__, avoid real client setup
    orch.planner = MagicMock()
    orch.coder = MagicMock()
    orch.critic = MagicMock()
    orch.max_revisions = max_revisions
    return orch


def test_approves_on_first_attempt():
    orch = build_orchestrator_with_mocks()
    orch.planner.plan.return_value = Plan(steps=["step1", "step2"], reasoning="simple task")
    orch.coder.implement.return_value = CodeSubmission(code="def f(): return 1", explanation="done")
    orch.critic.review.return_value = Review(approved=True, feedback="looks good")

    result = orch.run("do something")

    assert result.success is True
    assert result.revisions_used == 1
    assert result.final_code == "def f(): return 1"
    orch.coder.implement.assert_called_once()


def test_revises_then_approves():
    orch = build_orchestrator_with_mocks()
    orch.planner.plan.return_value = Plan(steps=["step1"], reasoning="reasoning")

    bad_submission = CodeSubmission(code="broken", explanation="attempt 1")
    good_submission = CodeSubmission(code="fixed", explanation="attempt 2")
    orch.coder.implement.side_effect = [bad_submission, good_submission]

    orch.critic.review.side_effect = [
        Review(approved=False, feedback="missing edge case for empty input"),
        Review(approved=True, feedback="looks good now"),
    ]

    result = orch.run("do something")

    assert result.success is True
    assert result.revisions_used == 2
    assert result.final_code == "fixed"
    assert orch.coder.implement.call_count == 2

    # second coder call should have received the prior code + feedback for self-correction
    second_call_kwargs = orch.coder.implement.call_args_list[1]
    assert second_call_kwargs.args[2] == "broken"  # prior_code
    assert "empty input" in second_call_kwargs.args[3]  # feedback


def test_gives_up_after_max_revisions():
    orch = build_orchestrator_with_mocks(max_revisions=2)
    orch.planner.plan.return_value = Plan(steps=["step1"], reasoning="reasoning")
    orch.coder.implement.return_value = CodeSubmission(code="still broken", explanation="nope")
    orch.critic.review.return_value = Review(approved=False, feedback="still wrong")

    result = orch.run("do something")

    assert result.success is False
    assert result.revisions_used == 2
    assert result.final_code == "still broken"  # last attempt returned, not silently discarded
    assert orch.coder.implement.call_count == 2
    assert any(e.action == "gave_up" for e in result.trace)


def test_trace_records_every_handoff():
    orch = build_orchestrator_with_mocks()
    orch.planner.plan.return_value = Plan(steps=["step1"], reasoning="reasoning")
    orch.coder.implement.return_value = CodeSubmission(code="code", explanation="done")
    orch.critic.review.return_value = Review(approved=True, feedback="good")

    result = orch.run("task")

    agents_in_trace = [e.agent for e in result.trace]
    assert agents_in_trace == ["Planner", "Coder", "Critic"]
