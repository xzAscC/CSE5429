from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    text: str
    category: str
    id: str
    expected_min_tokens: int = 10


_PROMPTS: list[Prompt] = [
    # short_factual (4)
    Prompt(
        id="sf_01",
        category="short_factual",
        text="What is the capital of Japan?",
    ),
    Prompt(
        id="sf_02",
        category="short_factual",
        text="Who wrote the novel 1984?",
    ),
    Prompt(
        id="sf_03",
        category="short_factual",
        text="What is the chemical symbol for gold?",
    ),
    Prompt(
        id="sf_04",
        category="short_factual",
        text="In which year did World War II end?",
    ),
    # medium_creative (4)
    Prompt(
        id="mc_01",
        category="medium_creative",
        text=(
            "Write a short paragraph describing a sunset over the ocean. "
            "Use vivid imagery and sensory details to paint a picture in the reader's mind."
        ),
    ),
    Prompt(
        id="mc_02",
        category="medium_creative",
        text=(
            "Compose a haiku about the feeling of debugging code late at night. "
            "Capture the frustration and eventual relief."
        ),
    ),
    Prompt(
        id="mc_03",
        category="medium_creative",
        text=(
            "Describe an imaginary city of the future where technology and nature coexist "
            "in harmony. What does a typical street look like?"
        ),
    ),
    Prompt(
        id="mc_04",
        category="medium_creative",
        text=(
            "Write a brief dialogue between two AI systems discussing the meaning of "
            "consciousness. Keep it philosophical but accessible."
        ),
    ),
    # long_reasoning (4)
    Prompt(
        id="lr_01",
        category="long_reasoning",
        text=(
            "Consider the following scenario: A company has 100 employees. "
            "60% work remotely, 30% work in a hybrid model, and the rest work on-site. "
            "If the company decides to require all employees to come to the office at least "
            "2 days a week, how many employees will need to change their work arrangement? "
            "Explain your reasoning step by step."
        ),
    ),
    Prompt(
        id="lr_02",
        category="long_reasoning",
        text=(
            "A farmer has a rectangular field that is twice as long as it is wide. "
            "The perimeter of the field is 300 meters. What are the dimensions of the field? "
            "Show your calculation process clearly."
        ),
    ),
    Prompt(
        id="lr_03",
        category="long_reasoning",
        text=(
            "In a study, researchers found that people who drink coffee have a 20% lower "
            "risk of a certain disease. However, coffee drinkers in the study also tended to "
            "exercise more. Explain why this is a confounding variable and how it affects the "
            "interpretation of the results."
        ),
    ),
    Prompt(
        id="lr_04",
        category="long_reasoning",
        text=(
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take "
            "100 machines to make 100 widgets? Think carefully about the rate of production "
            "per machine."
        ),
    ),
    # code_generation (3)
    Prompt(
        id="cg_01",
        category="code_generation",
        text=(
            "Write a Python function called 'is_palindrome' that takes a string as input "
            "and returns True if the string is a palindrome, False otherwise. The function "
            "should ignore spaces and be case-insensitive."
        ),
    ),
    Prompt(
        id="cg_02",
        category="code_generation",
        text=(
            "Implement a function in Python that finds the two largest numbers in a given "
            "list of integers. The function should return a tuple of the two numbers in "
            "descending order. Handle edge cases like empty lists."
        ),
    ),
    Prompt(
        id="cg_03",
        category="code_generation",
        text=(
            "Write a Python function that takes a list of dictionaries representing students "
            "with 'name' and 'grade' keys, and returns the names of students who scored above "
            "the average grade."
        ),
    ),
    # edge_case (3)
    Prompt(
        id="ec_01",
        category="edge_case",
        text="Yes.",
        expected_min_tokens=1,
    ),
    Prompt(
        id="ec_02",
        category="edge_case",
        text="Continue the pattern: 2, 4, 8, 16, 32,",
    ),
    Prompt(
        id="ec_03",
        category="edge_case",
        text="Repeat the word 'hello' ten times in a row, separated by spaces.",
    ),
    # repetition_sensitive (2)
    Prompt(
        id="rs_01",
        category="repetition_sensitive",
        text=("What is 1+1? What is 2+2? What is 3+3? What is 4+4? What is 5+5?"),
    ),
    Prompt(
        id="rs_02",
        category="repetition_sensitive",
        text="List the first 20 prime numbers in order.",
    ),
]


def get_all_prompts() -> list[Prompt]:
    return list(_PROMPTS)


def get_prompts_by_category(category: str) -> list[Prompt]:
    return [p for p in _PROMPTS if p.category == category]


def get_all_categories() -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for p in _PROMPTS:
        if p.category not in seen:
            seen.add(p.category)
            result.append(p.category)
    return result


def get_prompt_by_id(prompt_id: str) -> Prompt | None:
    for p in _PROMPTS:
        if p.id == prompt_id:
            return p
    return None
