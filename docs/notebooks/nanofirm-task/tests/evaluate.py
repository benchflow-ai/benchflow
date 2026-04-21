"""Evaluate contract risk analysis quality."""
import json
import sys

ANALYSIS_PATH = "/app/analysis.json"
REWARD_PATH = "/logs/verifier/reward.txt"

def evaluate():
    try:
        with open(ANALYSIS_PATH) as f:
            analysis = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FAIL: Cannot read analysis.json: {e}")
        return 0.0

    score = 0.0
    max_score = 5.0

    # Check risks exist and have required fields
    risks = analysis.get("risks", [])
    if len(risks) >= 3:
        score += 1.0
        valid_risks = sum(1 for r in risks if all(k in r for k in ("clause", "severity", "issue", "recommendation")))
        if valid_risks >= 3:
            score += 0.5
    else:
        print(f"FAIL: Need >= 3 risks, got {len(risks)}")

    # Check compound risks
    compound = analysis.get("compound_risks", [])
    if len(compound) >= 1:
        score += 1.0
        if all(k in compound[0] for k in ("clauses", "severity", "issue", "recommendation")):
            score += 0.5
    else:
        print("FAIL: Need >= 1 compound risk")

    # Check deal breakers
    breakers = analysis.get("deal_breakers", [])
    if len(breakers) >= 1:
        score += 0.5
    else:
        print("FAIL: Need >= 1 deal breaker")

    # Check summary
    summary = analysis.get("summary", "")
    if len(summary) > 20:
        score += 0.5

    reward = score / max_score
    print(f"Score: {score}/{max_score} = {reward:.2f}")
    return reward

if __name__ == "__main__":
    reward = evaluate()
    with open(REWARD_PATH, "w") as f:
        f.write(f"{reward}\n")
