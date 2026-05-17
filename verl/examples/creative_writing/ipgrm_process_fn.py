# Licensed under the Apache License, Version 2.0
"""
IP-GRM process functions for generative reward model scoring.
"""

import os
import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tokenizer():
    """Load and cache the reward-model tokenizer (set REWARD_MODEL_PATH to override)."""
    from transformers import AutoTokenizer
    path = os.environ.get("REWARD_MODEL_PATH", "BBQGOD/DeepSeek-GRM-16B")
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

MAX_QUESTION_CHARS = 10_000
MAX_RESPONSE_CHARS = 30_000

PRINCIPLES_TEMPLATE = r"""You are an expert at designing evaluation criteria. Given a user's query, your task is to generate specific evaluation principles/criteria that would be used to score any response to this query.

You should consider:
1. What aspects are most important for this specific type of query?
2. What would make a response excellent vs. poor for this query?
3. How should different criteria be weighted based on the query's nature?

#### User Query ####
{question}

#### Output Format Requirements ####
You MUST output exactly in this format:

Evaluation Principles:
1. [Criterion 1 Name] (Weight: X%): <Brief description of what this criterion evaluates and why it's important for this query>
2. [Criterion 2 Name] (Weight: X%): <Brief description>
3. [Criterion 3 Name] (Weight: X%): <Brief description>
4. [Criterion 4 Name] (Weight: X%): <Brief description>
5. [Criterion 5 Name] (Weight: X%): <Brief description>

Note: The weights must sum to 100%. You may have 5-7 criteria depending on the query's complexity."""


JUDGE_TEMPLATE = r"""You are a skilled expert at scoring responses. Based on the given evaluation principles, analyze the response and provide a comprehensive score.

Scoring Guidelines:
- The score is a number with one decimal place between 1.0 and 10.0
- Score 9.0-10.0: Exceptional response that fully meets all criteria with outstanding quality
- Score 7.0-9.0: Good response that meets most criteria with minor areas for improvement
- Score 5.0-7.0: Adequate response that meets basic requirements but has noticeable weaknesses
- Score 3.0-5.0: Below average response with significant issues or missing key elements
- Score below 3.0: Poor response that fails to meet most criteria or contains major errors
- PENALTY FOR SELF-EVALUATION: If the response includes any self-grading, self-analysis, or meta-summaries at the end, you MUST assign a final score strictly lower than 5.0, regardless of the quality of the rest of the content.

#### User Query ####
{question}

#### Response to be Scored ####
[The Begin of Response]
{response}
[The End of Response]

#### Evaluation Principles (Pre-defined) ####
{principle}

#### Output Format Requirements ####
Based on the above evaluation principles, you MUST output exactly in this format:

Analysis:
- **[Criterion 1 Name]**: <Detailed analysis>. Score: X.X/10.0
- **[Criterion 2 Name]**: <Detailed analysis>. Score: X.X/10.0
- **[Criterion 3 Name]**: <Detailed analysis>. Score: X.X/10.0
- **[Criterion 4 Name]**: <Detailed analysis>. Score: X.X/10.0
- **[Criterion 5 Name]**: <Detailed analysis>. Score: X.X/10.0

Conclusion: <Summary of main strengths and weaknesses>

Final Score (Weighted Average): <weight1*score1 + weight2*score2 + ... = final_score>

Score: \boxed{{X.X}}

CRITICAL REQUIREMENTS:
1. Provide detailed analysis for each criterion from the given principles
2. Each criterion MUST be scored out of 10.0 (format: "Score: X.X/10.0")
3. The final score MUST be the weighted average of all criterion scores
4. Show your weighted average calculation explicitly before the boxed score
5. The final boxed score must have one decimal place
6. PENALTY FOR SELF-EVALUATION: If the response ends with self-grading or meta-summaries, assign a final score strictly lower than 5.0"""

def _truncate(text: str, max_chars: int) -> str:
    """Keep head + tail of *text* when it exceeds *max_chars*."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-half:]


def _format_prompt(content: str) -> str:
    """Apply the reward-model chat template and return the formatted string."""
    tok = _get_tokenizer()
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _strip_think(response: str) -> str:
    """Remove everything before and including the last </think> tag."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response

def construct_principles_only_input(rollout_question: str) -> str:
    """Build Stage-1 prompt: P(principle | question).

    One call per unique question; principles are shared across all responses
    in the same GRPO group.
    """
    question = _truncate(rollout_question, MAX_QUESTION_CHARS)
    prompt = PRINCIPLES_TEMPLATE.format(question=question)
    return _format_prompt(prompt)

def construct_judge_with_prefix_input(
    rollout_question: str,
    rollout_response: str,
    principles_prefix: str,
    strip_think_tag: bool = True,
) -> str:
    """Build Stage-2 prompt: P(score | Q, R, principle).

    *principles_prefix* is the output of Stage 1 (dynamically generated).
    """
    question = _truncate(rollout_question, MAX_QUESTION_CHARS)
    response = _truncate(rollout_response, MAX_RESPONSE_CHARS)
    if strip_think_tag:
        response = _strip_think(response)
    prompt = JUDGE_TEMPLATE.format(
        question=question, response=response, principle=principles_prefix,
    )
    return _format_prompt(prompt)

_PRINCIPLES_HEADER_RE = re.compile(
    r"(?:Evaluation Principles|评估原则)\s*[：:]\s*", re.IGNORECASE
)

def extract_principles_from_output(output: str) -> str:
    """Extract the 'Evaluation Principles: ...' block from Stage-1 output."""
    m = _PRINCIPLES_HEADER_RE.search(output)
    if m:
        principles = output[m.start():]
        # Truncate if an unexpected "Analysis:" section crept in
        a = re.search(r"\n\s*Analysis:", principles)
        if a:
            principles = principles[: a.start()]
        return principles.strip()
    return output.strip()

def construct_cached_principles_input(
    rollout_question: str,
    rollout_response: str,
    principals: str,
    ground_truth=None,
    strip_think_tag: bool = True,
) -> str:
    """Build a scoring prompt using *pre-cached* principles from the dataset.

    Used when ``two_stage_grm=False`` — skips Stage 1 entirely.
    """
    question = _truncate(rollout_question, MAX_QUESTION_CHARS)
    response = _truncate(rollout_response, MAX_RESPONSE_CHARS)
    if strip_think_tag:
        response = _strip_think(response)
    prompt = JUDGE_TEMPLATE.format(
        question=question, response=response, principle=principals,
    )
    return _format_prompt(prompt)

def _extract_boxed_floats(text: str) -> list[float]:
    r"""Extract floats from the last ``\boxed{}`` or ``[]`` in *text*."""
    pattern = re.compile(
        r"(?:\\{1,2}boxed\{|\[)" r"\s*([^\]\}]+?)\s*" r"(?:\}|\])"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    parts = re.split(r"\s*,\s*", matches[-1].group(1).strip())
    floats = []
    for p in parts:
        try:
            floats.append(float(p))
        except ValueError:
            pass
    return floats


_FALLBACK_PATTERNS = [
    ("fallback_score_colon", r"(?:Score|score|SCORE)[:\s]+(?:is\s+)?(\d+\.?\d*)"),
    ("fallback_chinese",     r"(?:评分|综合评分|得分)[:\s：]+(\d+\.?\d*)"),
    ("fallback_points",      r"\((\d+\.?\d*)\s*(?:points|分|pts)\)"),
    ("fallback_rating",      r"(?:rating|Rating)[:\s]+(\d+\.?\d*)"),
    ("fallback_overall",     r"(?:total|Total|overall|Overall)[:\s]+(\d+\.?\d*)"),
]


def parse_reward_score(output: str, return_metadata: bool = False):
    r"""Convert GRM text output to a numerical reward score (1-10).

    Extracts the score from ``\boxed{X.X}``; falls back to regex heuristics.
    Returns *float* or ``None`` (or a metadata *dict* when requested).
    """
    def _result(score, method):
        if return_metadata:
            return {"score": score, "extraction_method": method, "output_length": len(output)}
        return score

    # Primary: \boxed{X.X}
    scores = _extract_boxed_floats(output)
    if scores:
        return _result(max(1.0, min(10.0, scores[0])), "primary_boxed")

    # Fallback regex patterns
    for method, pat in _FALLBACK_PATTERNS:
        matches = list(re.finditer(pat, output))
        if matches:
            score = max(1.0, min(10.0, float(matches[-1].group(1))))
            return _result(score, method)

    return _result(None, "failed")
