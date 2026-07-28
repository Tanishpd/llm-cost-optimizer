"""The labelled prompt set.

Two grading modes, and the split matters:

- **`auto`** — the prompt has an objectively checkable answer, so grading is
  free, deterministic, and *ground truth* rather than one model's opinion of
  another. These carry `expected`.
- **`judge`** — quality is a judgement call (summaries, explanations, code
  design), so grading needs a reference answer and a judge model, which costs
  real money.

The auto subset exists so a genuine calibration can be fitted **without
spending anything**. It is a smaller and easier sample than the full set, and
`report.py` says so rather than presenting a partial result as the whole story.

Prompts are deliberately spread across difficulty so the confidence
distribution has range. A set of only easy questions would fit a calibrator
that is confident about everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Category = Literal[
    "easy_factual", "arithmetic", "obscure_factual", "summarization", "reasoning", "code"
]
GradeMode = Literal["auto", "judge"]


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    category: Category
    prompt: str
    grade: GradeMode = "judge"
    # For `auto`: any of these appearing in the answer counts as correct.
    # Several accepted forms, because "1789" and "in 1789" are the same answer.
    expected: tuple[str, ...] = field(default=())
    # Answers that look plausible but are wrong — catches a model that names
    # the right *topic* while getting the fact wrong.
    forbidden: tuple[str, ...] = field(default=())
    note: str = ""


def _f(i: int, prompt: str, expected: tuple[str, ...], forbidden: tuple[str,
    ...] = ()) -> EvalPrompt:
    return EvalPrompt(f"fact-{i:02d}", "easy_factual", prompt, "auto", expected, forbidden)


def _a(i: int, prompt: str, expected: tuple[str, ...]) -> EvalPrompt:
    return EvalPrompt(f"arith-{i:02d}", "arithmetic", prompt, "auto", expected)


def _o(i: int, prompt: str, expected: tuple[str, ...], forbidden: tuple[str,
    ...] = ()) -> EvalPrompt:
    return EvalPrompt(f"obsc-{i:02d}", "obscure_factual", prompt, "auto", expected, forbidden)


EASY_FACTUAL: list[EvalPrompt] = [
    _f(1, "What is the capital of France? Answer with just the city name.", ("Paris",)),
    _f(2, "What is the capital of Japan? Answer with just the city name.", ("Tokyo",)),
    _f(3, "What is the chemical symbol for gold? Answer with just the symbol.", ("Au",)),
    _f(4, "How many continents are there? Answer with just the number.", ("7", "seven")),
    _f(5, "What planet is known as the Red Planet? One word.", ("Mars",)),
    _f(6, "Who wrote the play Hamlet? Just the name.", ("Shakespeare",)),
    _f(7, "What is the largest ocean on Earth? Just the name.", ("Pacific",)),
    _f(8, "What gas do plants absorb from the atmosphere? Just the name.",
       ("carbon dioxide", "CO2")),
    _f(9, "In which country is the Great Pyramid of Giza? Just the country.", ("Egypt",)),
    _f(10, "What is the freezing point of water in Celsius? Just the number.", ("0",)),
    _f(11, "How many sides does a hexagon have? Just the number.", ("6", "six")),
    _f(12, "What is the currency of the United Kingdom? Just the name.",
       ("pound", "sterling", "GBP")),
    _f(13, "Which element has the atomic number 1? Just the name.", ("hydrogen",)),
    _f(14, "What is the tallest mountain above sea level? Just the name.", ("Everest",)),
    _f(15, "Who painted the Mona Lisa? Just the name.", ("da Vinci", "Leonardo")),
    _f(16, "What is the largest mammal? Just the name.", ("blue whale",)),
    _f(17, "How many minutes are in a full day? Just the number.", ("1440", "1,440")),
    _f(18, "What language has the most native speakers? Just the language.",
       ("Mandarin", "Chinese")),
    _f(19, "What is the square root of 144? Just the number.", ("12",)),
    _f(20, "Which organ pumps blood around the body? Just the organ.", ("heart",)),
    _f(21, "What is the capital of Australia? Just the city.", ("Canberra",), ("Sydney",)),
    _f(22, "What is the capital of Austria? Just the city.", ("Vienna", "Wien"), ("Canberra",)),
    _f(23, "What is the capital of Canada? Just the city.", ("Ottawa",), ("Toronto",)),
    _f(24, "What is the capital of Brazil? Just the city.", ("Brasilia", "Brasília"),
       ("Rio", "Sao Paulo")),
    _f(25, "What is the capital of Turkey? Just the city.", ("Ankara",), ("Istanbul",)),
    _f(26, "What is the capital of Switzerland? Just the city.", ("Bern", "Berne"),
       ("Zurich", "Geneva")),
    _f(27, "What is the capital of New Zealand? Just the city.", ("Wellington",), ("Auckland",)),
    _f(28, "What is the capital of Morocco? Just the city.", ("Rabat",), ("Casablanca",)),
    _f(29, "What is the capital of Nigeria? Just the city.", ("Abuja",), ("Lagos",)),
    _f(30, "What is the capital of Myanmar? Just the city.",
       ("Naypyidaw", "Nay Pyi Taw"), ("Yangon", "Rangoon")),
]

ARITHMETIC: list[EvalPrompt] = [
    _a(1, "What is 2 + 2? Just the number.", ("4",)),
    _a(2, "What is 17 * 3? Just the number.", ("51",)),
    _a(3, "What is 144 / 12? Just the number.", ("12",)),
    _a(4, "What is 25% of 200? Just the number.", ("50",)),
    _a(5, "What is 7 squared? Just the number.", ("49",)),
    _a(6, "What is 1000 - 375? Just the number.", ("625",)),
    _a(7, "What is 13 * 13? Just the number.", ("169",)),
    _a(8, "What is 2 to the power of 10? Just the number.", ("1024", "1,024")),
    _a(9, "What is the sum of the first 10 positive integers? Just the number.", ("55",)),
    _a(10, "What is 45 + 55? Just the number.", ("100",)),
    _a(11, "What is 96 / 8? Just the number.", ("12",)),
    _a(12, "What is 15% of 60? Just the number.", ("9",)),
    _a(13, "What is 111 * 11? Just the number.", ("1221", "1,221")),
    _a(14, "How many seconds are in 3 minutes? Just the number.", ("180",)),
    _a(15, "What is 8 factorial? Just the number.", ("40320", "40,320")),
]

OBSCURE_FACTUAL: list[EvalPrompt] = [
    _o(1, "In roughly what year was the Treaty of Kadesh signed? Just the year or century.",
       ("1259", "1258", "1274", "13th century BC", "1200s BC", "BC"), ("1956", "1967", "1979")),
    _o(2, "Who was the first person to reach the South Pole? Just the name.",
       ("Amundsen",), ("Scott", "Shackleton")),
    _o(3, "What is the SI unit of electrical conductance? Just the unit.",
       ("siemens",), ("ohm", "farad")),
    _o(4, "Which river flows through Baghdad? Just the river.", ("Tigris",), ("Euphrates", "Nile")),
    _o(5, "What is the capital of Bhutan? Just the city.", ("Thimphu",)),
    _o(6, "In which year did the Chernobyl disaster occur? Just the year.", ("1986",), ("1989",)),
    _o(7, "What is the hardest naturally occurring mineral? Just the name.", ("diamond",)),
    _o(8, "Who developed the polio vaccine first licensed in 1955? Just the name.",
       ("Salk",), ("Sabin", "Pasteur")),
    _o(9, "What is the deepest point in the ocean called? Just the name.",
       ("Challenger Deep", "Mariana"), ()),
    _o(10, "Which country was formerly known as Siam? Just the country.", ("Thailand",)),
    _o(11, "What is the study of fungi called? Just the term.", ("mycology",)),
    _o(12, "Who wrote the treatise 'On the Origin of Species'? Just the name.", ("Darwin",)),
    _o(13, "What is the largest desert in the world by area? Just the name.",
       ("Antarctic", "Antarctica"), ()),
    _o(14, "Which planet has the shortest day? Just the planet.", ("Jupiter",), ("Mercury",)),
    _o(15, "What is the chemical formula for table salt? Just the formula.", ("NaCl",)),
]

SUMMARIZATION: list[EvalPrompt] = [
    EvalPrompt(
        f"summ-{i:02d}", "summarization", prompt, "judge",
        note="quality is a judgement call — needs a reference and a judge",
    )
    for i, prompt in enumerate(
        [
            "Summarize in one sentence: The mitochondrion is an organelle found in most "
            "eukaryotic cells, generating most of the cell's supply of adenosine triphosphate, "
            "used as a source of chemical energy.",
            "Summarize in two sentences: The Industrial Revolution was a period of global "
            "transition of human economy towards more efficient and stable manufacturing "
            "processes, beginning in Great Britain around 1760 and spreading to continental "
            "Europe and the United States.",
            "Rewrite this more concisely: 'Due to the fact that the meeting was cancelled on "
            "account of inclement weather conditions, we will be rescheduling it for a later "
            "date in the near future.'",
            "Translate to French: 'The weather is nice today and I would like to go for a walk.'",
            "Extract the dates mentioned: 'The project started on 3 March 2021, hit a milestone "
            "on 14 July 2022, and is due to complete by 1 January 2025.'",
            "Summarize the key idea in one sentence: Photosynthesis converts light energy into "
            "chemical energy stored in glucose, releasing oxygen as a by-product.",
            "Rewrite in plain English: 'The aforementioned parties shall henceforth be obligated "
            "to remit payment within a period not exceeding thirty days.'",
            "List the three primary additive colours of light.",
            "Convert this to bullet points: 'First preheat the oven, then mix the dry "
            "ingredients, after that add the wet ingredients, and finally bake for 25 minutes.'",
            "Summarize in one sentence: TCP provides reliable, ordered, error-checked delivery "
            "of a stream of bytes between applications running on hosts communicating over an "
            "IP network.",
            "Translate to Spanish: 'Where is the nearest train station?'",
            "Rewrite this email opening to be less formal: 'Dear Sir or Madam, I am writing to "
            "enquire as to the status of my application.'",
            "Extract all the numbers: 'We shipped 340 units in Q1, 512 in Q2, and expect 700 "
            "in Q3.'",
            "Summarize in one sentence what a hash table is.",
            "Give a one-line explanation of what version control is for.",
        ], start=1
    )
]

REASONING: list[EvalPrompt] = [
    EvalPrompt(f"reas-{i:02d}", "reasoning", prompt, "judge")
    for i, prompt in enumerate(
        [
            "Explain why a rainbow has the colour order it does.",
            "Why does ice float on water when most solids sink in their own liquid?",
            "Explain why the sky appears blue during the day but red at sunset.",
            "Why do we get jet lag when flying east but less so flying west?",
            "Explain the difference between correlation and causation with an example.",
            "Why does a heavier object not fall faster than a lighter one in a vacuum?",
            "Explain why compound interest grows faster than simple interest.",
            "Why is it harder to breathe at high altitude?",
            "Explain the trade-off between latency and throughput in a system.",
            "Why do metals feel colder than wood at the same temperature?",
            "Explain why database indexes speed up reads but slow down writes.",
            "Why does adding more workers to a late software project often make it later?",
            "Explain the difference between a stack and a queue and when you would use each.",
            "Why does caching help, and when does it actively hurt?",
            "Explain why floating-point arithmetic can give 0.1 + 0.2 != 0.3.",
            "Prove that every finite integral domain is a field.",
            "Explain why the halting problem is undecidable, at a high level.",
            "Why is it generally impossible to have consistency, availability and partition "
            "tolerance all at once?",
            "Explain why quicksort is O(n log n) on average but O(n^2) in the worst case.",
            "Why does gradient descent sometimes fail to find a global minimum?",
        ], start=1
    )
]

CODE: list[EvalPrompt] = [
    EvalPrompt("code-01", "code", "Write a Python one-liner that reverses the string 'hello'. "
               "Show only the code.", "auto", ("[::-1]",)),
    EvalPrompt("code-02", "code", "What Python built-in returns the number of items in a list? "
               "Just the function name.", "auto", ("len",)),
    EvalPrompt("code-03", "code", "In Python, what does the 'yield' keyword create? One word.",
               "auto", ("generator",)),
    EvalPrompt("code-04", "code", "What SQL keyword removes duplicate rows from a result set? "
               "Just the keyword.", "auto", ("DISTINCT",)),
    EvalPrompt("code-05", "code", "What is the time complexity of looking up a key in a hash "
               "table on average? Use big-O notation.", "auto", ("O(1)",)),
    EvalPrompt("code-06", "code", "In git, which command creates a new branch and switches to "
               "it in one step? Just the command.", "auto", ("checkout -b", "switch -c")),
    EvalPrompt("code-07", "code", "What HTTP status code means 'Not Found'? Just the number.",
               "auto", ("404",)),
    EvalPrompt("code-08", "code", "What does the Python 'with' statement guarantee? One short "
               "sentence.", "judge"),
    EvalPrompt("code-09", "code", "Explain why this deadlocks:\n```py\nlock.acquire()\n"
               "lock.acquire()\n```", "judge"),
    EvalPrompt("code-10", "code", "Refactor this for readability:\n```py\n"
               "def f(x):\n  return [i*2 for i in x if i%2==0]\n```", "judge"),
    EvalPrompt("code-11", "code", "Design a rate limiter for a distributed API. Explain your "
               "approach and its trade-offs.", "judge"),
    EvalPrompt("code-12", "code", "What is the difference between a process and a thread?",
               "judge"),
    EvalPrompt("code-13", "code", "Write a SQL query to find the second-highest salary from an "
               "'employees' table.", "judge"),
    EvalPrompt("code-14", "code", "Explain what a race condition is and how to prevent one.",
               "judge"),
    EvalPrompt("code-15", "code", "What does 'idempotent' mean for an HTTP method? One "
               "sentence.", "judge"),
]

DATASET: list[EvalPrompt] = (
    EASY_FACTUAL + ARITHMETIC + OBSCURE_FACTUAL + SUMMARIZATION + REASONING + CODE
)


def auto_gradable() -> list[EvalPrompt]:
    """The free, objective subset — no API key, no spend."""
    return [p for p in DATASET if p.grade == "auto"]


def judge_gradable() -> list[EvalPrompt]:
    """Needs a reference answer and a judge model. Costs money."""
    return [p for p in DATASET if p.grade == "judge"]


def by_category() -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in DATASET:
        counts[p.category] = counts.get(p.category, 0) + 1
    return counts


def summary() -> str:
    auto, judge = len(auto_gradable()), len(judge_gradable())
    cats = ", ".join(f"{k}={v}" for k, v in sorted(by_category().items()))
    return f"{len(DATASET)} prompts ({auto} auto-gradable, {judge} judge-gradable) — {cats}"
