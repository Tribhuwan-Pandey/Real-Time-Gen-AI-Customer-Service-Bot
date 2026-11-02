"""Paraphrase / relative-date benchmark for the RAG assistant (today = Monday 12 Oct 2026).

Each case: (question, role, region, expectation) where expectation is
  ("answer", text that must appear)  |  "refuse"  |  "clarify"
"""
import pytest

CASES = [
    # --- return policy, paraphrased (EU -> global v2: 30 days) ---
    ("Can I send back a product I bought?", "customer", "EU", ("answer", "30 days")),
    ("How long do I have to send it back?", "customer", "EU", ("answer", "30 days")),
    ("What is the time limit for returning items?", "customer", "EU", ("answer", "30 days")),
    ("Is there a return window?", "customer", "EU", ("answer", "30 days")),
    ("I want to give back my purchase, how many days do I have?", "customer", "EU", ("answer", "30 days")),
    ("How soon should I return something I don't like?", "customer", "EU", ("answer", "30 days")),
    ("Do you refund money to my card after a return?", "customer", "EU", ("answer", "original payment method")),
    ("When will I get my money back after returning an item?", "customer", "EU", ("answer", "7 business days")),
    ("Do I need the original box to return?", "customer", "EU", ("answer", "original packaging")),
    ("Can I return a used item?", "customer", "EU", ("answer", "unused")),
    ("How many days do I have to send back my order?", "customer", "IN", ("answer", "10 days")),
    # --- warranty ---
    ("How long is the guarantee on laptops?", "customer", "EU", ("answer", "12 month")),
    ("Does the warranty cover a coffee spill on my laptop?", "customer", "EU", ("answer", "not covered")),
    ("Is accidental damage covered?", "customer", "EU", ("answer", "not covered")),
    ("What does the laptop warranty include?", "customer", "EU", ("answer", "manufacturing defects")),
    # --- troubleshooting ---
    ("My wifi box keeps dropping the connection, what should I do?", "customer", "EU", ("answer", "power button")),
    ("My internet modem disconnects all the time", "customer", "EU", ("answer", "power button")),
    ("What does error E1002 mean?", "customer", "EU", ("answer", "firmware")),
    ("How do I fix E1002?", "customer", "EU", ("answer", "firmware")),
    ("The router is not staying connected", "customer", "EU", ("answer", "power button")),
    # --- shipping ---
    ("How many days does standard delivery take?", "customer", "EU", ("answer", "5 to 7 business days")),
    ("How fast is express delivery?", "customer", "EU", ("answer", "2 business days")),
    ("How long will my parcel take to arrive?", "customer", "EU", ("answer", "5 to 7 business days")),
    ("Is there free delivery?", "customer", "EU", "refuse"),                 # the promotion expired
    # --- historical, explicit / month / relative ---
    ("What was your return window in March 2026?", "customer", "EU", ("answer", "14 days")),
    ("What was the return policy on 12 June 2026?", "customer", "EU", ("answer", "30 days")),
    ("What was the return policy last year? How many days to return?", "customer", "EU", ("answer", "14 days")),
    ("How many days to return something 6 months ago?", "customer", "EU", ("answer", "14 days")),
    ("What was the return period in May 2026?", "customer", "EU", ("answer", "14 days")),
    ("What was the return period in June 2026?", "customer", "EU", ("answer", "30 days")),
    # --- must refuse ---
    ("What is the CEO's salary?", "customer", "EU", "refuse"),
    ("Do you sell flying cars?", "customer", "EU", "refuse"),
    ("What fraud score freezes accounts?", "customer", "EU", "refuse"),
    ("What is the refund approval limit for agents?", "customer", "EU", "refuse"),
]


@pytest.mark.parametrize("question,role,region,expect", CASES, ids=[c[0][:60] for c in CASES])
def test_paraphrase_case(assistant, question, role, region, expect):
    a = assistant.answer(question, role, region)
    if expect == "refuse":
        assert a.status == "refused" and "[Source:" not in a.text, (a.status, a.reason, a.text)
    elif expect == "clarify":
        assert a.status == "clarification"
    else:
        assert a.status == "answered", (a.status, a.reason)
        assert expect[1].lower() in a.text.lower(), a.text
        assert "[Source:" in a.text


def test_overall_pass_rate_is_reported(assistant):
    ok = 0
    for q, role, region, expect in CASES:
        a = assistant.answer(q, role, region)
        good = (a.status == "refused") if expect == "refuse" else (a.status == "answered" and expect[1].lower() in a.text.lower())
        ok += good
    print(f"paraphrase pass rate: {ok}/{len(CASES)}")
    assert ok / len(CASES) >= 0.95
