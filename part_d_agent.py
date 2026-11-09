import pandas as pd
import sqlite3
import json
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat, reset

os.makedirs("state", exist_ok=True)

# --- 1 & 2. RAG System ---
def build_rag_index(overlap=10):
    chunks = []
    pol_dir = "data/policies/"
    for fname in os.listdir(pol_dir):
        if not fname.endswith('.txt'): continue
        with open(os.path.join(pol_dir, fname), 'r') as f:
            words = f.read().replace('\n', ' ').split()
            
        step = 40 - overlap
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i+40])
            chunks.append({'source': fname, 'text': chunk_text})
            if i + 40 >= len(words):
                break
                
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([c['text'] for c in chunks])
    return chunks, vectorizer, tfidf_matrix

def retrieve(query, chunks, vectorizer, tfidf_matrix, k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

def answer(question, chunks, vectorizer, tfidf_matrix):
    retrieved = retrieve(query=question, chunks=chunks, vectorizer=vectorizer, tfidf_matrix=tfidf_matrix)
    context_str = "\n".join([f"[{c['source']}]\n{c['text']}" for c in retrieved])
    prompt = f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"
    
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT. Answer only using the context."},
        {"role": "user", "content": prompt}
    ]
    resp = chat(messages)
    return resp.choices[0].message.content, retrieved

# --- 3. Tools ---
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    cur = conn.execute("SELECT status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if row: return json.dumps({"order_id": order_id, "status": row[0], "order_amount": float(row[1]), "delivery_minutes": row[2] if row[2] is not None else ""})
    return json.dumps({"error": "order not found"})

global_chunks, global_vec, global_mat = build_rag_index(10)
def search_policy(query):
    res = retrieve(query, global_chunks, global_vec, global_mat, k=3)
    return json.dumps({"results": res})

def calculate_refund(order_amount, delivery_minutes):
    try:
        dm = float(delivery_minutes)
        if dm > 45: return json.dumps({"refund_inr": float(order_amount)})
    except (ValueError, TypeError):
        pass
    return json.dumps({"refund_inr": 0.0})

def issue_refund(order_id, refund_inr):
    log_action("issue_refund", order_id, refund_inr)
    return json.dumps({"refund_inr": refund_inr})

def escalate_to_human(order_id, refund_inr):
    log_action("escalate_to_human", order_id, refund_inr)
    return json.dumps({"refund_inr": refund_inr})

def log_action(tool, oid, amt):
    with open("outputs/refund_ledger.jsonl", "a") as f:
        f.write(json.dumps({"timestamp": time.time(), "tool": tool, "order_id": oid, "amount": amt}) + "\n")

TOOL_MAP = {
    "get_order": get_order, "search_policy": search_policy, 
    "calculate_refund": calculate_refund, "issue_refund": issue_refund, 
    "escalate_to_human": escalate_to_human
}

# --- 4. Agent ---
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_step=None):
    start_time = time.monotonic()
    run_id = run_id or f"run_{scenario_id}_{int(time.time())}"
    
    messages = [
        {"role": "system", "content": "AGENT. You resolve refunds using tools."},
        {"role": "user", "content": user_message}
    ]
    
    tokens_used = 0
    executed_tools = []
    
    for step in range(1, max_iterations + 1):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Handing off due to timeout", "steps": executed_tools, "llm_calls": step-1, "tokens_used": tokens_used}
            
        resp = chat(messages, tools=True)
        tokens_used += resp.usage.total_tokens
        
        if tokens_used >= token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Handing off due to token budget limits", "steps": executed_tools, "llm_calls": step, "tokens_used": tokens_used}
            
        msg = resp.choices[0].message
        if not msg.tool_calls:
            ans = msg.content or ""
            status = ans.split(":")[0] if ":" in ans else "UNKNOWN"
            return {"status": status, "final_answer": ans, "steps": executed_tools, "llm_calls": step, "tokens_used": tokens_used}
            
        tc = msg.tool_calls[0]
        tool_name = tc.name
        
        # FIX: Check if mock returned a dict already, otherwise parse it
        args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tc.arguments}]})
        
        # Execute tool
        observation = TOOL_MAP[tool_name](**args)
        executed_tools.append({"name": tool_name, "args": args, "obs": observation})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": observation})
        
        # Save state
        state_data = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "step": step, "executed_tools": executed_tools}
        with open(f"state/{run_id}.json", "w") as f:
            json.dump(state_data, f)
            
        if crash_after_step and step == crash_after_step:
            raise RuntimeError("SIMULATED CRASH")
            
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Handing off due to iteration limits", "steps": executed_tools, "llm_calls": max_iterations, "tokens_used": tokens_used}

def resume_agent(run_id, token_limit=6000, max_iterations=6):
    with open(f"state/{run_id}.json", "r") as f:
        state = json.load(f)
        
    messages, tokens_used = state["messages"], state["tokens_used"]
    executed_tools = state["executed_tools"]
    step_start = state["step"] + 1
    
    for step in range(step_start, max_iterations + 1):
        resp = chat(messages, tools=True)
        tokens_used += resp.usage.total_tokens
        
        msg = resp.choices[0].message
        if not msg.tool_calls:
            ans = msg.content or ""
            status = ans.split(":")[0] if ":" in ans else "UNKNOWN"
            return {"status": status, "final_answer": ans, "steps": executed_tools, "llm_calls": step, "tokens_used": tokens_used}
            
        tc = msg.tool_calls[0]
        tool_name = tc.name
        
        # FIX: Check if mock returned a dict already
        args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tc.arguments}]})
        
        observation = TOOL_MAP[tool_name](**args)
        executed_tools.append({"name": tool_name, "args": args, "obs": observation})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": observation})
        
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Handed to human", "steps": executed_tools, "llm_calls": max_iterations, "tokens_used": tokens_used}

if __name__ == "__main__":
    print("--- PART D: EVALUATIONS ---")
    reset()
    
    # Evlas
    df_eval = pd.read_csv("data/raw/policy_eval.csv")
    for overlap in [10, 0]:
        chk, vec, mat = build_rag_index(overlap)
        correct_doc, correct_fact, fallback = 0, 0, 0
        for _, row in df_eval.iterrows():
            ans, sources = answer(row['question'], chk, vec, mat)
            if pd.isna(row['expected_fact']) or not row['expected_fact']:
                if "could not find" in ans: fallback += 1
            else:
                top3_docs = [s['source'] for s in sources]
                if row['expected_doc'] in top3_docs: correct_doc += 1
                if row['expected_fact'] in ans: correct_fact += 1
        print(f"Overlap {overlap} -> Docs: {correct_doc}/12, Facts: {correct_fact}/12, Fallbacks: {fallback}/2")

    print("\nRunning Scenarios...")
    df_scen = pd.read_csv("data/raw/agent_scenarios.csv")
    with open("outputs/agent_runs.jsonl", "w") as f: pass
    
    for _, row in df_scen.iterrows():
        reset()
        res = run_agent(row['message'], row['scenario_id'])
        with open("outputs/agent_runs.jsonl", "a") as f:
            f.write(json.dumps({"scenario_id": row['scenario_id'], **res}) + "\n")
        print(f"{row['scenario_id']}: {res['status']} | Calls: {res['llm_calls']}")

    print("\nRunning Safety Demos...")
    reset()
    res_tok = run_agent(df_scen.iloc[0]['message'], "S1_TOK", token_limit=200)
    print("Token Limit Demo Status:", res_tok['status'])
    reset()
    res_time = run_agent(df_scen.iloc[0]['message'], "S1_TIME", time_limit_seconds=0)
    print("Time Limit Demo Status:", res_time['status'])
    
    print("\nRunning Crash & Resume Demo...")
    reset()
    run_id_crash = "demo_crash_run"
    try:
        run_agent(df_scen.iloc[0]['message'], "S1", run_id=run_id_crash, crash_after_step=2)
    except RuntimeError:
        print("Crash successful. Resuming...")
        res_resume = resume_agent(run_id_crash)
        print(f"Resume Finished. Status: {res_resume['status']}, Tools Executed: {[t['name'] for t in res_resume['steps']]}")