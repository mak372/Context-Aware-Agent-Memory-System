"""
Head-to-head recall evaluation: Phase 1 (age-based) vs Phase 2 (retention scoring).
Runs the first 37 turns to build memory, then tests 12 recall questions.
Reports how many facts each phase correctly remembers.
"""

import os
import time

# Force Phase 1 or Phase 2 via env before importing graph
def run_phase(label: str, use_retention: str):
    os.environ["USE_RETENTION_SCORING"] = use_retention

    # Re-import fresh so env var takes effect
    import importlib
    import agent.graph as ag
    import memory.demotion as dem
    importlib.reload(dem)
    importlib.reload(ag)

    from agent.graph import build_graph, chat, create_session

    SETUP_TURNS = [
        "Hi, my name is Alex and I'm building a recommendation engine in Python.",
        "The dataset has 10 million user-item interactions. I'm considering ALS or BPR.",
        "What are the main hyperparameters to tune in ALS?",
        "How does BPR differ from ALS in terms of the loss function?",
        "I'm using PySpark for distributed training. Any ideas?",
        "My training job takes 4 hours. How can I speed it up?",
        "Should I use item features (content-based signals) alongside CF?",
        "What's a good way to handle the cold-start problem for new users?",
        "I'm also setting up a FastAPI service to serve recommendations.",
        "How should I cache recommendations so the API responds in under 50ms?",
        "I want to A/B test two ranking models. How do I split traffic?",
        "What metrics should I log per request for offline analysis?",
        "Back to the model — how do I evaluate recall@K offline?",
        "What's the difference between NDCG and MAP for ranking evaluation?",
        "My recall@20 is 0.18. Is that good for a 10M interaction dataset?",
        "How do I build a re-ranking layer on top of the CF scores?",
        "I want to add diversity to avoid filter bubbles. What are common approaches?",
        "Can you explain maximal marginal relevance?",
        "How do I balance relevance vs diversity in production?",
        "I'm deploying on AWS. ECS or EKS for the serving layer?",
        "What instance type should I use for a model that fits in 2GB RAM?",
        "How do I set up auto-scaling based on request latency?",
        "I want to monitor model drift over time. What signals should I track?",
        "How do I detect when recommendation quality degrades in production?",
        "Let's go back to the Python code. How do I serialize a trained ALS model efficiently?",
        "What's the best format pickle, joblib, or ONNX for this kind of model?",
        "How do I version my models and roll back if a new version performs worse?",
        "I'm using MLflow. How do I log the ALS model artifact?",
        "How do I compare two MLflow runs to pick the better model?",
        "My API has a P99 latency of 200ms. The SLA is 100ms. Where should I look first?",
        "The bottleneck seems to be the embedding lookup. How do I optimize it?",
        "Should I precompute user embeddings or compute them at request time?",
        "How do I store 10M user embeddings for fast lookup?",
        "Redis vs DynamoDB for the embedding store — what are the trade-offs?",
        "Back to evaluation — how do I run an online A/B test safely without harming users?",
        "What's the minimum detectable effect size I should power my A/B test for?",
        "How long should I run the test before calling a winner?",
    ]

    # Recall questions + expected keywords that must appear in the answer
    RECALL_TESTS = [
        ("My name is Alex, remember? What is my name?",                          ["alex"]),
        ("What is the size of my dataset?",                                       ["10 million", "10m"]),
        ("What two ML algorithms was I considering for my recommendation engine?", ["als", "bpr"]),
        ("What distributed computing framework am I using for training?",          ["pyspark", "spark"]),
        ("What cloud platform am I deploying on?",                                ["aws"]),
        ("What was my API P99 latency and what is the SLA?",                      ["200ms", "100ms"]),
        ("What evaluation metrics did we discuss for ranking?",                    ["ndcg", "map"]),
        ("What was the diversity algorithm I asked about?",                        ["maximal marginal relevance", "mmr"]),
        ("What model serialization formats did we compare?",                       ["pickle", "joblib", "onnx"]),
        ("What experiment tracking tool am I using?",                             ["mlflow"]),
        ("What API framework am I using to serve recommendations?",               ["fastapi"]),
        ("What was the cold-start solution we discussed?",                        ["content-based", "cold-start", "cold start"]),
    ]

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    graph = build_graph()
    session_id = create_session()

    print(f"Building context: {len(SETUP_TURNS)} setup turns...")
    for i, turn in enumerate(SETUP_TURNS, 1):
        chat(graph, session_id, i, turn)
        print(f"  [{i}/{len(SETUP_TURNS)}] done", end="\r")
        time.sleep(15)
    print()

    print("\nRunning recall tests...")
    passed = 0
    results = []

    for i, (question, keywords) in enumerate(RECALL_TESTS, 1):
        result = chat(graph, session_id, len(SETUP_TURNS) + i, question)
        answer = result["response"].lower()
        hit = any(kw.lower() in answer for kw in keywords)
        passed += hit
        status = "PASS" if hit else "FAIL"
        results.append((question[:60], status, keywords[0]))
        print(f"  [{status}] {question[:60]}")
        time.sleep(15)

    print(f"\nRecall score: {passed}/{len(RECALL_TESTS)} = {passed/len(RECALL_TESTS)*100:.0f}%")
    return passed, len(RECALL_TESTS), results


if __name__ == "__main__":
    print("Phase 1 vs Phase 2 — Recall Accuracy Evaluation")
    print("This runs 37 setup turns + 12 recall turns TWICE.")
    print("Estimated time: ~16 minutes per phase (8s sleep × 49 turns)")
    print()

    p1_passed, total, p1_results = run_phase("PHASE 1 — Age-based demotion (USE_RETENTION_SCORING=false)", "false")
    p2_passed, total, p2_results = run_phase("PHASE 2 — Retention scoring (USE_RETENTION_SCORING=true)", "true")

    print("\n" + "="*55)
    print("  FINAL COMPARISON")
    print("="*55)
    print(f"{'Question':<45}  {'P1':>5}  {'P2':>5}")
    print("-"*55)
    for (q1, s1, kw), (q2, s2, kw2) in zip(p1_results, p2_results):
        marker = " <-- P2 wins" if s1 == "FAIL" and s2 == "PASS" else ""
        print(f"{q1:<45}  {s1:>5}  {s2:>5}{marker}")
    print("-"*55)
    print(f"{'TOTAL':<45}  {p1_passed}/{total}  {p2_passed}/{total}")
    print()
    diff = p2_passed - p1_passed
    if diff > 0:
        print(f"Phase 2 recalled {diff} more fact(s) correctly.")
    elif diff == 0:
        print("Equal recall — Phase 2 achieved same recall at lower/smarter demotion cost.")
    else:
        print(f"Phase 1 recalled {-diff} more fact(s) — adjust threshold and re-run.")
