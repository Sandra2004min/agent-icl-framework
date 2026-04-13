"""Test Adapters (QA, Math, Code)"""

import sys
sys.path.insert(0, '../src')

from icl_agent.adapters import QAAdapter, MathAdapter, CodeAdapter


# ─── QAAdapter Tests ───

def test_qa_evaluate_exact_match():
    qa = QAAdapter()
    score = qa.evaluate({"answer": "42"}, {"answer": "42"})
    assert score == 1.0
    print("PASS: qa_exact_match")


def test_qa_evaluate_case_insensitive():
    qa = QAAdapter()
    score = qa.evaluate({"answer": "Friday"}, {"answer": "friday"})
    assert score == 1.0
    print("PASS: qa_case_insensitive")


def test_qa_evaluate_number_extraction():
    qa = QAAdapter()
    score = qa.evaluate({"answer": "The answer is 10"}, {"answer": "10"})
    assert score == 1.0
    print("PASS: qa_number_extraction")


def test_qa_evaluate_partial_match():
    qa = QAAdapter()
    score = qa.evaluate({"answer": "yes it is friday"}, {"answer": "friday"})
    assert score == 0.5
    print("PASS: qa_partial_match")


def test_qa_evaluate_no_match():
    qa = QAAdapter()
    score = qa.evaluate({"answer": "banana"}, {"answer": "42"})
    assert score == 0.0
    print("PASS: qa_no_match")


def test_qa_execute_mock():
    qa = QAAdapter(llm_client=None)
    output = qa.execute({"system_prompt": "test"}, {"question": "What is 2+2?"})
    assert "answer" in output
    print("PASS: qa_execute_mock")


# ─── MathAdapter Tests ───

def test_math_evaluate_gsm8k_format():
    ma = MathAdapter()
    score = ma.evaluate({"answer": "The answer is 42"}, {"answer": "#### 42"})
    assert score == 1.0
    print("PASS: math_gsm8k_format")


def test_math_evaluate_exact_number():
    ma = MathAdapter()
    score = ma.evaluate({"answer": "#### 18"}, {"answer": "#### 18"})
    assert score == 1.0
    print("PASS: math_exact_number")


def test_math_evaluate_comma_dollar():
    ma = MathAdapter()
    score = ma.evaluate({"answer": "Total is $1,000"}, {"answer": "#### 1000"})
    assert score == 1.0
    print("PASS: math_comma_dollar")


def test_math_evaluate_boxed():
    ma = MathAdapter()
    score = ma.evaluate({"answer": "\\boxed{42}"}, {"answer": "#### 42"})
    assert score == 1.0
    print("PASS: math_boxed")


def test_math_evaluate_wrong():
    ma = MathAdapter()
    score = ma.evaluate({"answer": "The answer is 99"}, {"answer": "#### 42"})
    assert score == 0.0
    print("PASS: math_wrong")


def test_math_evaluate_float_tolerance():
    ma = MathAdapter()
    score = ma.evaluate({"answer": "3.14"}, {"answer": "#### 3.14"})
    assert score == 1.0
    print("PASS: math_float_tolerance")


def test_math_execute_mock():
    ma = MathAdapter(llm_client=None)
    output = ma.execute({"system_prompt": "test"}, {"question": "What is 6*7?"})
    assert "answer" in output
    print("PASS: math_execute_mock")


# ─── CodeAdapter Tests ───

def test_code_evaluate_exact_match():
    ca = CodeAdapter()
    score = ca.evaluate(
        {"answer": "Initialize result to 1 instead of 0"},
        {"answer": "Initialize result to 1 instead of 0"}
    )
    assert score == 1.0
    print("PASS: code_exact_match")


def test_code_evaluate_keyword_match():
    ca = CodeAdapter()
    score = ca.evaluate(
        {"answer": "You should initialize result to 1 instead of 0 to fix the factorial"},
        {"answer": "result should be initialized to 1 instead of 0"}
    )
    assert 0.0 < score <= 1.0
    print("PASS: code_keyword_match (score={:.2f})".format(score))


def test_code_evaluate_empty():
    ca = CodeAdapter()
    score = ca.evaluate({"answer": ""}, {"answer": "fix the bug"})
    assert score == 0.0
    print("PASS: code_evaluate_empty")


def test_code_evaluate_code_block():
    ca = CodeAdapter()
    predicted = "Here is the fix:\n```python\nresult = 1\n```"
    expected = "```python\nresult = 1\n```"
    score = ca.evaluate({"answer": predicted}, {"answer": expected})
    assert score >= 0.8
    print("PASS: code_code_block (score={:.2f})".format(score))


def test_code_execute_mock():
    ca = CodeAdapter(llm_client=None)
    output = ca.execute({"system_prompt": "test"}, {"question": "Fix the bug"})
    assert "answer" in output
    print("PASS: code_execute_mock")


if __name__ == "__main__":
    print("Running Adapter tests...\n")

    print("--- QAAdapter ---")
    test_qa_evaluate_exact_match()
    test_qa_evaluate_case_insensitive()
    test_qa_evaluate_number_extraction()
    test_qa_evaluate_partial_match()
    test_qa_evaluate_no_match()
    test_qa_execute_mock()

    print("\n--- MathAdapter ---")
    test_math_evaluate_gsm8k_format()
    test_math_evaluate_exact_number()
    test_math_evaluate_comma_dollar()
    test_math_evaluate_boxed()
    test_math_evaluate_wrong()
    test_math_evaluate_float_tolerance()
    test_math_execute_mock()

    print("\n--- CodeAdapter ---")
    test_code_evaluate_exact_match()
    test_code_evaluate_keyword_match()
    test_code_evaluate_empty()
    test_code_evaluate_code_block()
    test_code_execute_mock()

    print("\nAll Adapter tests passed!")
