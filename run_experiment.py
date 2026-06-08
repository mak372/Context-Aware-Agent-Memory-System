"""
Run a scripted 50-turn conversation through the tiered memory agent
and compare token usage against the naive (append-everything) approach.
"""

from agent.graph import build_graph, chat, create_session
from measurement.tracker import NaiveTracker, TurnRecord, print_summary, save_results

# Scripted conversation — 50 turns covering varied topics so cold retrieval is exercised
TURNS = [
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
    "My name is Alex, remember? I want to make sure you still have context about my project.",
    "Can you summarize what my project is about based on our conversation so far?",
    "What tech stack have I mentioned so far?",
    "What were the main model choices I was considering?",
    "What infrastructure am I deploying on?",
    "What was my P99 latency issue?",
    "What evaluation metrics did we discuss?",
    "How many interactions are in my dataset?",
    "What was the cold-start solution we talked about?",
    "What's the name of the diversity algorithm I asked about?",
    "What model serialization formats did we compare?",
    "What was the turn where I asked about A/B testing traffic splitting?",
    "Give me a final summary of all the key decisions I've made in this conversation.",
]

assert len(TURNS) == 50, f"Expected 50 turns, got {len(TURNS)}"


def main():
    print("Initializing session...")
    graph = build_graph()
    session_id = create_session()
    naive = NaiveTracker()
    records: list[TurnRecord] = []

    for i, user_input in enumerate(TURNS, start=1):
        print(f"\n[Turn {i}/50] {user_input[:60]}...")

        result = chat(graph, session_id, i, user_input)
        tiered_tokens = result["tokens_used"]
        naive_tokens = naive.add_turn(user_input, result["response"])

        records.append(TurnRecord(turn=i, tiered_tokens=tiered_tokens, naive_tokens=naive_tokens))
        print(f"  Tiered: {tiered_tokens} tokens | Naive: {naive_tokens} tokens")
        print(f"  Response: {result['response'][:100]}...")

    print("\n" + "=" * 50)
    print_summary(records)
    save_results(records)


if __name__ == "__main__":
    main()
