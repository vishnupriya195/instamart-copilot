import pandas as pd
import re
import json
import random
import time
from mock_llm import chat, TransientLLMError, est_tokens, reset

# Configuration
VP_TICKETS_PATH = "data/raw/tickets.csv"
VP_SEGMENTS_PATH = "outputs/customer_segments.csv"
VP_ALLOWED_CATS = {'billing', 'delivery', 'technical', 'fraud'}
BASE_DELAY = 0.05
MAX_INPUT_TOKENS = 400

def redact_pii(text):
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    text = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text = re.sub(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text

def resilient_llm_call(messages, token_budget=None, current_usage=0):
    for n in range(3):
        try:
            resp = chat(messages)
            if token_budget and (current_usage + resp.usage.total_tokens) >= token_budget:
                return None, resp.usage.total_tokens, True # True means budget exhausted
            return resp, resp.usage.total_tokens, False
        except TransientLLMError:
            wait_time = BASE_DELAY * (2**n) * random.uniform(0.9, 1.15)
            print(f"Retry {n} for ticket. Waiting {wait_time:.3f}s")
            time.sleep(wait_time)
    return None, 0, False

def validate_llm_output(content):
    try:
        data = json.loads(content)
        if data.get('category') in VP_ALLOWED_CATS:
            return data['category'], 'classified_json'
    except json.JSONDecodeError:
        pass
    
    words = re.findall(r"[a-z]+", str(content).lower())
    found_cats = [w for w in words if w in VP_ALLOWED_CATS]
    if len(found_cats) == 1:
        return found_cats[0], 'classified_keyword'
        
    return None, 'flagged_invalid_output'

def run_triage_pipeline(prompt_type, token_budget=None):
    df_tickets = pd.read_csv(VP_TICKETS_PATH, keep_default_na=False)
    
    # Load segments for routing
    try:
        df_segments = pd.read_csv(VP_SEGMENTS_PATH)
        premium_customers = set(df_segments[df_segments['is_premium'] == True]['customer_id'])
        segment_map = df_segments.set_index('customer_id')['segment_name'].to_dict()
    except:
        premium_customers = set()
        segment_map = {}

    system_prompt = "CLASSIFY support tickets into exactly one of: billing, delivery, technical, fraud. Reply in strict JSON format: {\"category\": \"...\", \"reason\": \"...\"}."
    if prompt_type == "few-shot":
        system_prompt += "\nEXAMPLES:\nTicket: \"Someone hacked my account.\" -> {\"category\": \"fraud\", \"reason\": \"hacked account\"}\nTicket: \"App crashes.\" -> {\"category\": \"technical\", \"reason\": \"app crash\"}"

    reset() # Reset mock counters before full run
    
    results = []
    total_tokens = 0
    pii_counts = {'email': 0, 'phone': 0, 'order': 0}
    
    for _, row in df_tickets.iterrows():
        tid, cid, raw_text = row['ticket_id'], row['customer_id'], str(row['text'])
        
        if not raw_text.strip():
            results.append({'ticket_id': tid, 'customer_id': cid, 'status': 'skipped_empty', 'category': '', 'priority': 'NORMAL', 'route': 'human_review', 'tokens_used': 0})
            continue
            
        if est_tokens(raw_text) > MAX_INPUT_TOKENS:
            results.append({'ticket_id': tid, 'customer_id': cid, 'status': 'flagged_long', 'category': '', 'priority': 'NORMAL', 'route': 'human_review', 'tokens_used': 0})
            continue

        pii_counts['email'] += len(re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', raw_text))
        pii_counts['phone'] += len(re.findall(r'\b[6-9]\d{9}\b', raw_text))
        pii_counts['order'] += len(re.findall(r'ORD-\d{6}', raw_text))
        
        safe_text = redact_pii(raw_text)
        assert len(re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', safe_text)) == 0, "PII leak!"
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": safe_text}]
        
        resp, tokens, budget_hit = resilient_llm_call(messages, token_budget, total_tokens)
        
        if budget_hit:
            results.append({'ticket_id': tid, 'customer_id': cid, 'status': 'budget_exhausted', 'category': '', 'priority': 'NORMAL', 'route': 'human_review', 'tokens_used': 0})
            break
            
        total_tokens += tokens
        cat, status = validate_llm_output(resp.choices[0].message.content)
        
        is_prem = cid in premium_customers
        seg_name = segment_map.get(cid, "")
        
        # Routing Logic
        if status == 'flagged_invalid_output':
            route, priority = 'human_review', 'NORMAL'
        elif cat == 'fraud':
            route, priority = 'security_team', 'HIGH'
        elif is_prem:
            route, priority = 'premium_desk', 'HIGH'
        else:
            route, priority = 'support_queue', 'NORMAL'
            
        results.append({
            'ticket_id': tid, 'customer_id': cid, 'status': status, 'category': cat or '', 
            'segment_name': seg_name, 'priority': priority, 'route': route, 'tokens_used': tokens
        })

    df_out = pd.DataFrame(results)
    
    # Calculate accuracy over classified
    classified_df = df_out[df_out['status'].isin(['classified_json', 'classified_keyword'])].copy()
    if len(classified_df) > 0:
        merged = classified_df.merge(df_tickets[['ticket_id', 'true_category']], on='ticket_id')
        acc = (merged['category'] == merged['true_category']).mean()
    else:
        acc = 0
        
    return df_out, acc, total_tokens, pii_counts

if __name__ == "__main__":
    print("--- PART C: TRIAGE PIPELINE ---")
    # 1. Demo Token Budget
    print("Running Zero-Shot Budget Demo...")
    demo_df, _, _, _ = run_triage_pipeline("zero-shot", token_budget=5000)
    demo_df.to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"Exhausted demo saved. Budget hit status count: {(demo_df['status'] == 'budget_exhausted').sum()}")
    
    # 2. Compare Zero vs Few Shot
    print("\nRunning Zero-Shot Full...")
    zs_df, zs_acc, zs_tok, pii = run_triage_pipeline("zero-shot", token_budget=100000)
    zs_class_count = zs_df['status'].isin(['classified_json', 'classified_keyword']).sum()
    print(f"Zero-Shot -> Classified: {zs_class_count}, Acc: {zs_acc:.2%}, Total Tok: {zs_tok}, Tok/Tick: {zs_tok/max(1, zs_class_count):.1f}")
    print(f"PII Redacted: {pii}")
    
    print("\nRunning Few-Shot Full...")
    fs_df, fs_acc, fs_tok, _ = run_triage_pipeline("few-shot", token_budget=100000)
    fs_class_count = fs_df['status'].isin(['classified_json', 'classified_keyword']).sum()
    print(f"Few-Shot -> Classified: {fs_class_count}, Acc: {fs_acc:.2%}, Total Tok: {fs_tok}, Tok/Tick: {fs_tok/max(1, fs_class_count):.1f}")
    
    # Save Final output
    fs_df.to_csv("outputs/triage_results.csv", index=False)
    print("Saved final few-shot routing to outputs/triage_results.csv")