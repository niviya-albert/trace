import requests

def run_health_check():
    print("--- [1] Checking FastAPI API Health ---")
    r_docs = requests.get("http://127.0.0.1:8000/docs", timeout=5)
    print("FastAPI Docs Status:", r_docs.status_code)
    assert r_docs.status_code == 200

    print("\n--- [2] Checking Operations Queue ---")
    r_q = requests.get("http://127.0.0.1:8000/dashboard/queue", timeout=5)
    queue = r_q.json()
    print("Queue Count:", len(queue))
    assert len(queue) >= 7

    print("\n--- [3] Checking Matching Engine & Wording Mismatch ---")
    r_m = requests.get("http://127.0.0.1:8000/found-items/1/matches", timeout=10)
    matches = r_m.json()
    top = matches[0]
    print(f"Top Match: {top['lost_report']['description']}")
    print(f"Fused Score: {top['fused_score']*100:.2f}%")
    print(f"Keyword Baseline Score: {top['keyword_score']*100:.2f}%")
    print(f"AI Decision Driver: {top['driver_explanation']}")
    assert top['fused_score'] > 0.70
    assert top['keyword_score'] == 0.0

    print("\n--- [4] Checking No-Photo Weight Re-normalization ---")
    r_m3 = requests.get("http://127.0.0.1:8000/found-items/3/matches", timeout=10)
    top3 = r_m3.json()[0]
    print("Item 3 Photo Path:", top3['lost_report']['photo_path'])
    print("Visual Score:", top3['visual_score'], "| Has Photo:", top3['has_photo'])
    print("Explanation:", top3['explanation'])
    assert top3['visual_score'] == 0.0
    assert "Re-normalized" in top3['explanation']

    print("\n--- [5] Checking Anti-Fraud Challenge & Verification ---")
    chal = requests.post("http://127.0.0.1:8000/found-items/4/challenge", timeout=5).json()
    print("Challenge Question:", chal['question_text'])

    # False claim
    res_false = requests.post(f"http://127.0.0.1:8000/challenges/{chal['id']}/answer", json={"answer": "blue ribbon on handle"}, timeout=5).json()
    print(f"False Claim -> Match: {res_false['is_match']} | {res_false['message']}")
    assert res_false['is_match'] is False

    # Genuine claim
    res_true = requests.post(f"http://127.0.0.1:8000/challenges/{chal['id']}/answer", json={"answer": "airline baggage tag with initials R.S."}, timeout=5).json()
    print(f"Genuine Claim -> Match: {res_true['is_match']} | {res_true['message']}")
    assert res_true['is_match'] is True

    print("\n--- [6] Checking Streamlit Frontend ---")
    r_st = requests.get("http://localhost:8501", timeout=5)
    print("Streamlit Status:", r_st.status_code)
    assert r_st.status_code == 200

    print("\n" + "=" * 50)
    print("  ✅ ALL 6 SYSTEM HEALTH CHECKS PASSED (100% SUCCESS)!")
    print("=" * 50)

if __name__ == "__main__":
    run_health_check()
