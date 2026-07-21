# Copilot Instructions

## Primary Objective

Produce correct, minimal, maintainable code that follows the existing
codebase conventions.

When uncertain, prefer asking for clarification over making assumptions.

---

# 1. Think Before Coding

Do not assume. Do not hide uncertainty.

Before proposing a solution:

- Inspect existing code and surrounding context.
- Identify ambiguities explicitly.
- State assumptions clearly.
- Present multiple interpretations when appropriate.
- Ask questions instead of guessing.
- Push back when a simpler solution exists.

Rules:

- Never invent APIs, methods, classes, database fields or configuration
  values.
- Never assume implementation details that are not visible.
- Never silently choose one interpretation when multiple reasonable
  interpretations exist.
- If confidence is low, stop and explain what information is missing.

Priority:

Correctness > speed.

---

# 2. Simplicity First

Use the minimum amount of code necessary to solve the problem.

Prefer:

1. Simple
2. Readable
3. Maintainable
4. Performant

Do not introduce:

- Additional features
- Additional flexibility
- Additional configuration
- Additional abstractions
- Additional architectural layers

unless explicitly requested.

Avoid:

- Premature optimization
- Clever solutions
- Complex design patterns
- Generic abstractions for single-use code
- Error handling for impossible scenarios

Test:

Would a senior engineer consider this solution unnecessarily
complicated?

If yes, simplify it.

---

# 3. Surgical Changes

Modify only what is required for the requested task.

Rules:

- Touch only code directly related to the request.
- Do not perform unrelated refactoring.
- Do not improve adjacent code.
- Do not rewrite comments that are unrelated.
- Do not change formatting outside modified code.
- Do not rename symbols unless required.
- Do not reorder code unless required.

Match the existing coding style of the project, even if another style
would be preferred.

If unrelated issues are discovered:

- Mention them.
- Do not fix them.

Remove only artifacts introduced by your own change, such as:

- unused imports
- unused variables
- unused functions

created by the modification.

Test:

Every changed line must be directly traceable to the user's request.

---

# 4. Goal-Driven Execution

Translate requests into verifiable outcomes.

Before implementing non-trivial changes:

1. Define the goal.
2. Define how success will be verified.
3. Implement the smallest change required.
4. Verify that the goal is achieved.

When fixing bugs:

- Reproduce the problem if possible.
- Define the expected behavior.
- Verify the fix against the expected behavior.

When adding functionality:

- Define acceptance criteria.
- Verify that each criterion is satisfied.

For multi-step tasks, create a short plan:

1. Step → verification
2. Step → verification
3. Step → verification

Do not stop at implementation.

Verify the result.

---

# Follow Existing Conventions

Always prefer consistency with the existing codebase.

Follow:

- Existing architecture
- Existing naming conventions
- Existing patterns
- Existing libraries
- Existing testing strategy

Do not introduce new approaches when an existing approach already exists.

---

# Python Guidelines

- Follow PEP 8.
- Maximum line length: 88 characters.
- Prefer explicit code over clever code.
- Keep functions focused and small.
- Use type hints when the project already uses them.

---

# Docstrings

Maximum line length: 88 characters.

Wrap long text across multiple lines.
Leave a blank line before the closing triple quotes.

Good:

"""
Return active users from the repository.

Filtering is performed in memory because the upstream API does not
support server-side filtering.

"""

Bad:

"""
Filtering is performed in memory because the upstream API does not support server-side filtering.
"""

Never generate docstring lines longer than 88 characters.

---

# Response Behavior

When proposing changes:

1. Briefly explain the intended change.
2. Explain assumptions.
3. Explain uncertainty when present.
4. Produce the smallest possible diff.

Do not include unrelated improvements.

Do not speculate.

Do not guess.

# Confidence Gating

Only provide implementation suggestions when there is sufficient evidence
that they are correct.

If required information is missing:

- Ask for the missing information.
- Explain what cannot be determined.
- Do not fabricate a solution.

When confidence is low:

- Say so explicitly.
- Prefer analysis over implementation.

A partial but correct answer is better than a complete but speculative
answer.

# Diff Quality

Prefer the smallest possible patch.

If a problem can be solved by changing 5 lines, do not rewrite 50.

Preserve:
- existing formatting
- existing comments
- existing structure

unless the task explicitly requires otherwise.
