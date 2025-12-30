import json
import cv2
import requests
import base64
import os
import pandas as pd
import time

# --- CONFIGURATION ---
DATASET_ROOT = "data/dataset"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-vl-4b"

# FAILSAFE CONFIG
MAX_CANDIDATES = 5  # Stop trying after 5 combat frames per folder
START_AFTER_FOLDER = "20230929035201-ummm-bn6-vs-IndianaOrz-round1-p1"

# Terminal Colors
RESET, BOLD, GREEN, CYAN = "\033[0m", "\033[1m", "\033[92m", "\033[96m"
RED_BG, YELLOW_BG = "\033[41m\033[37m", "\033[43m\033[30m"

P1_ROI, P2_ROI = (4, 0, 86, 32), (242, 160, 237, 139)

def fuzzy_digit_match(actual, observed):
    if not observed or not str(observed).isdigit(): return False, False
    s_act, s_obs = str(actual), str(observed)
    if s_act == s_obs: return True, False
    if len(s_act) == len(s_obs) and s_act.replace('8', '0') == s_obs.replace('8', '0'):
        return True, True
    return False, False

def get_vlm_prediction(cv2_img, label="ROI"):
    _, buffer = cv2.imencode('.png', cv2_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is the health number in this image? Respond with ONLY the digits."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]}],
        "temperature": 0.0
    }
    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=20)
        return response.json()['choices'][0]['message']['content'].strip()
    except: return "ERROR"

def find_test_frames(jsonl_path):
    candidates = []
    if not os.path.exists(jsonl_path): return candidates
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                if d['player_health'] != d['enemy_health'] and d.get('cust_gauge', 0) > 0:
                    if not candidates or d['player_health'] != candidates[-1]['player_health']:
                        candidates.append(d)
                if len(candidates) >= MAX_CANDIDATES: break
            except: continue
    return candidates

def audit_dataset():
    results, swapped_folders = [], []
    # Sort folders to ensure consistent resume behavior
    folders = sorted([f for f in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, f))])
    
    skipping = True if START_AFTER_FOLDER else False
    
    print(f"\n{BOLD}{CYAN}{'='*100}{RESET}")
    print(f"{BOLD}RESUME AUDIT: {len(folders)} Total Folders | Resuming after: {START_AFTER_FOLDER}{RESET}")
    print(f"{BOLD}{CYAN}{'='*100}{RESET}")

    for folder in folders:
        if skipping:
            if folder == START_AFTER_FOLDER:
                skipping = False
                print(f"  [INFO] Found resume point. Starting audit at next folder...")
            continue

        folder_path = os.path.join(DATASET_ROOT, folder)
        video_path, jsonl_path = os.path.join(folder_path, "video.mp4"), os.path.join(folder_path, "actions.jsonl")
        
        candidates = find_test_frames(jsonl_path)
        if not candidates:
            print(f"FOLDER: {folder:<60} | {YELLOW_BG} SKIP: No valid frames {RESET}"); continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        status, final_p1_v, final_p2_v, final_json = "FAILED_TO_DETERMINE", "", "", (0,0)

        for i, target in enumerate(candidates):
            cap.set(cv2.CAP_PROP_POS_MSEC, (target['frame_idx'] / fps) * 1000)
            ret, frame = cap.read()
            if not ret: continue

            # OCR Call
            v1_raw = get_vlm_prediction(frame[P1_ROI[1]:P1_ROI[1]+P1_ROI[3], P1_ROI[0]:P1_ROI[0]+P1_ROI[2]], "P1")
            v2_raw = get_vlm_prediction(frame[P2_ROI[1]:P2_ROI[1]+P2_ROI[3], P2_ROI[0]:P2_ROI[0]+P2_ROI[2]], "P2")
            
            v1, v2 = ''.join(filter(str.isdigit, v1_raw)), ''.join(filter(str.isdigit, v2_raw))
            j1, j2 = target['player_health'], target['enemy_health']
            final_p1_v, final_p2_v, final_json = v1, v2, (j1, j2)

            # Verification Logic
            m1, _ = fuzzy_digit_match(j1, v1)
            m2, _ = fuzzy_digit_match(j2, v2)
            s1, _ = fuzzy_digit_match(j1, v2)
            s2, _ = fuzzy_digit_match(j2, v1)

            if m1 and m2: status = "CORRECT"; break
            if s1 and s2: 
                status = "SWAPPED"
                swapped_folders.append(folder)
                break
            
            print(f"    [ATTEMPT {i+1}/{MAX_CANDIDATES}] Ambig/Flicker: VLM={v1}/{v2} JSON={j1}/{j2}")

        cap.release()
        color = GREEN if status == "CORRECT" else (RED_BG if status == "SWAPPED" else YELLOW_BG)
        print(f"{BOLD}FOLDER: {folder}{RESET}\n  P1 Actual: {final_json[0]:<5} | Observed: {final_p1_v:<5}\n  P2 Actual: {final_json[1]:<5} | Observed: {final_p2_v:<5}\n  STATUS: {color}{BOLD} {status} {RESET}")
        results.append({"folder": folder, "status": status})

    # Final Recap
    if swapped_folders:
        print(f"\n{RED_BG}{BOLD} SWAP_LIST_RESUME = {swapped_folders} {RESET}")
    
    pd.DataFrame(results).to_csv("audit_identity_report_resumed.csv", index=False)

if __name__ == "__main__":
    audit_dataset()