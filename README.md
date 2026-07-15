# Multi-Agent Orchestrator

Three agents — Planner, Coder, Critic — collaborating on a coding task,
with a bounded self-correction loop and a full visible trace of every
handoff between them.

## Why this exists

Most "multi-agent" demos are a single model talking to itself with
extra prompt scaffolding. The part worth demonstrating is the actual
coordination logic: when does control pass from one role to the next,
what information crosses that boundary, and what happens when a step
fails. This project makes all of that explicit and testable.

## Architecture

```
        Task
         │
         ▼
   ┌───────────┐
   │  Planner   │  breaks task into 3-6 ordered steps
   └─────┬─────┘
         │ plan
         ▼
   ┌───────────┐
   │   Coder    │◄──────────────┐
   └─────┬─────┘                │
         │ code                 │ feedback + prior code
         ▼                      │
   ┌───────────┐                │
   │  Critic    │────rejected───┘
   └─────┬─────┘   (bounded by max\\\_revisions)
         │ approved
         ▼
   Final code + full trace
```

Every arrow in this diagram is a recorded `TraceEvent` — after a run you
can print the full sequence of what each agent said and decided, not
just the final output.

## Design decisions (interview talking points)

1. **Bounded revision loop.** Critic rejection sends feedback back to
Coder, but only up to `max\\\_revisions` times. If it still isn't
approved, the orchestrator returns the last attempt clearly marked
`success=False` — it does not silently pretend the code is good.
2. **Structured handoffs, not prose.** Each agent returns a typed
dataclass (`Plan`, `CodeSubmission`, `Review`) parsed from strict
JSON, so the orchestrator never has to regex a decision out of free
text. This is the same pattern used in the dev-agent project, reused
deliberately.
3. **Full trace, not just final output.** `OrchestrationResult.trace`
records every agent action in order. This is what you'd show an
interviewer to prove the agents are actually coordinating, not just
one model called three times with different prompts.
4. **No framework.** Hand-rolled state machine (\~90 lines) instead of
LangGraph/CrewAI, so every line of coordination logic is something
you can explain and defend in an interview.

## Setup

Runs on [Groq](https://console.groq.com) — free API keys, and Groq's LPU
inference is extremely fast, so the Planner → Coder → Critic loop completes
in a couple seconds even with revisions. Default model is
`llama-3.3-70b-versatile`.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Groq API key (console.groq.com/keys)
export $(cat .env | xargs)
```

## Running it

```bash
python -m orchestrator.cli --task "Write a function that checks if a string is a palindrome, ignoring case and spaces"
```

## Running the tests

```bash
PYTHONPATH=src pytest tests/ -v
```

All 4 orchestration-logic tests (first-attempt approval, revise-then-approve,
give-up-after-max-revisions, full-trace-recorded) run with mocked agents —
zero API cost, verifying the state machine itself is correct independent of
model behavior.

## Example trace (real run, Groq / llama-3.3-70b-versatile)

```
Task: Write a function that checks if a string is a palindrome, ignoring case and spaces

Running Planner -> Coder -> Critic loop...

AGENT TRACE

\\\[Planner] created\\\_plan: 5 steps: To determine if a string is a palindrome, we need to preprocess it by removing spaces and ignoring case, then compare it with its reverse. This approach ensures the comparison is case-insensitive and space-ignoring.

\\\[Coder] submitted\\\_v1: This function works by first removing spaces and converting to lowercase, then comparing the resulting string with its reverse using Python's slice notation.

\\\[Critic] approved: Code is correct and effectively checks if a string is a palindrome, ignoring case and spaces, with a time complexity of O(n) due to string reversal and replacement operations.

✅ APPROVED after 1 attempt(s)

Final code:

def is\\\_palindrome(s):

s = s.replace(' ', '')

s = s.lower()

return s == s\\\[::-1]



Approved on the first attempt here — worth also running a task with trickier

edge cases (see Demo script below) to show the revision loop actually firing,

since that's the most interesting part of this project to explain live.



\\## Demo script (for a portfolio video)



1\\. Pick a task with a genuine edge case the first attempt is likely to

\&#x20;  miss (empty input, type mismatch, off-by-one) — this makes the

\&#x20;  revision loop actually fire, which is the interesting part to show.

2\\. Run the CLI live, narrate each agent's line as it prints.

3\\. Point out the moment the Critic rejects and explain what changed in

\&#x20;  the Coder's second attempt.Demo script (for a portfolio video)

1. Pick a task with a genuine edge case the first attempt is likely to
miss (empty input, type mismatch, off-by-one) — this makes the
revision loop actually fire, which is the interesting part to show.
2. Run the CLI live, narrate each agent's line as it prints.
3. Point out the moment the Critic rejects and explain what changed in
the Coder's second attempt.

## What I'd build next

\* A 4th agent (Tester) that actually executes the code against generated
test cases, rather than the Critic reviewing by reading alone.
\* Parallel exploration: have 2 Coders propose different approaches, Critic
picks the stronger one.
\* Persist trace history to disk so multi-session runs can be replayed.


