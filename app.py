"""
Meeting-Notes-to-Action-Items Summarizer — App (Use Case #9)
Transcript/text input, AI summary generation, action-item extraction (owner + due date), export to task list.
Works with Gemini 1.5 Flash free key, plus rule-based fallback so demo never breaks.
"""
import streamlit as st
import pandas as pd
import re
import json
import requests
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent

st.set_page_config(page_title="Meeting Summarizer", page_icon="📝", layout="wide")

ACTION_VERBS = ["will", "shall", "must", "need to", "needs to", "action", "todo", "assign", "deadline", "complete", "share", "send", "prepare", "draft", "review", "handle", "test", "collect", "create", "build"]
DECISION_WORDS = ["decided", "agreed", "final", "approved", "confirmed", "concluded"]
HIGH_WORDS = ["urgent", "urgently", "asap", "critical", "blocker", "high priority", "important"]
DATE_PATTERNS = [
    r"\d{1,2}\s?(Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep)[a-z]*(\s?\d{4})?",
    r"(tomorrow|today|tonight|EOD|end of (day|week))",
    r"by\s+(tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|EOD|\d{1,2}\s?\w+)",
    r"next\s+(monday|tuesday|wednesday|thursday|friday|week)",
    r"\d{4}-\d{2}-\d{2}",
]

def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]

def extract_owner(sent, attendees):
    # @Name pattern
    m = re.search(r"@(\w+)", sent)
    if m: return m.group(1).capitalize()
    # "Name will/should/shall" at start
    m = re.match(r"^(\w+)\s*[:,]?\s*(will|shall|should|must|agreed|noted)", sent, re.I)
    if m:
        cand = m.group(1).capitalize()
        if cand.lower() not in ("we", "team", "they", "it", "i"):
            return cand
    # "X will ..." anywhere with attendee match
    for a in attendees:
        if re.search(rf"\b{re.escape(a)}\b\s*(will|shall|should|must|agreed)", sent, re.I):
            return a
        if re.search(rf"\b{re.escape(a)}\b", sent):
            if any(v in sent.lower() for v in ACTION_VERBS):
                return a
    return "Unassigned"

def extract_due(sent):
    for pat in DATE_PATTERNS:
        m = re.search(pat, sent, re.I)
        if m: return m.group(0).strip()
    return "No date"

def priority_of(sent):
    s = sent.lower()
    if any(w in s for w in HIGH_WORDS): return "High"
    if any(w in s for w in ["should", "soon", "review", "please"]): return "Medium"
    return "Low"

def rule_based_process(text, attendees):
    sents = split_sentences(text)
    # Summary: score by keywords + decisions + length
    scored = []
    for i, s in enumerate(sents):
        sl = s.lower()
        score = 0
        score += sum(2 for w in DECISION_WORDS if w in sl)
        score += sum(1 for v in ACTION_VERBS if v in sl)
        score += 1 if any(re.search(p, s, re.I) for p in DATE_PATTERNS) else 0
        score += 0.5 if i < 3 else 0
        if 20 < len(s) < 220: score += 0.5
        scored.append((score, i, s))
    scored.sort(reverse=True)
    summary = [s for _, _, s in scored[:5]]
    # keep original order
    summary = sorted(summary, key=lambda x: text.find(x[:20]))
    decisions = [s for s in sents if any(w in s.lower() for w in DECISION_WORDS)][:6]
    actions = []
    for s in sents:
        sl = s.lower()
        if "attendees:" in sl and "will" not in sl:
            continue
        if any(v in sl for v in ACTION_VERBS) or re.search(r"\bwill\b", sl):
            owner = extract_owner(s, attendees)
            due = extract_due(s)
            # keep only meaningful
            if len(s) > 12:
                # clean "Name: ..." prefix for task
                task = re.sub(r"^\w+\s*:\s*", "", s).strip()
                actions.append({"task": task, "owner": owner, "due": due, "priority": priority_of(s), "source": "rule"})
    # de-dup
    seen, uniq = set(), []
    for a in actions:
        k = a["task"][:60].lower()
        if k not in seen:
            seen.add(k); uniq.append(a)
    return summary, decisions, uniq[:12]

def gemini_process(api_key, text, attendees, title):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
    prompt = f"""You are a meeting-notes assistant. Read this meeting transcript and return STRICT JSON only (no markdown fences).

Keys: summary (list of 4-6 short bullets), decisions (list of bullets, may be empty), actions (list of objects with task, owner, due, priority High/Medium/Low).

Rules: every action must have owner (use attendee names {attendees} or 'Unassigned') and due (exact phrase from text like 'by Friday', '3 October', or 'No date'). Keep tasks under 20 words. Meeting title: {title}.
Transcript:
{text[:6000]}"""
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code != 200:
            return None, f"Gemini API error {r.status_code}: {r.text[:300]}"
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        # strip fences if model adds them
        txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M)
        data = json.loads(txt)
        return data, None
    except Exception as e:
        return None, f"Gemini call failed: {e}"

def to_markdown(title, summary, decisions, actions_df):
    md = f"# Meeting Notes — {title}\n_Date: {datetime.now().strftime('%d %b %Y')}_\n\n## Summary\n"
    for s in summary: md += f"- {s}\n"
    md += "\n## Decisions\n"
    md += "\n".join(f"- {d}" for d in decisions) if decisions else "- None recorded\n"
    md += "\n\n## Action Items\n| # | Task | Owner | Due | Priority |\n|---|------|-------|-----|----------|\n"
    for i, r in actions_df.iterrows():
        md += f"| {i+1} | {r['task']} | {r['owner']} | {r['due']} | {r['priority']} |\n"
    md += "\n_Generated by Meeting Summarizer (Gemini + rule fallback). Verify owners/dates before sharing._\n"
    return md

# ---------- UI ----------
st.title("📝 Meeting-Notes-to-Action-Items Summarizer")
st.caption("App • Transcript input → AI summary + owner/due actions → export task list (Gemini free key + offline fallback)")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini Free API Key (optional)", type="password")
    st.markdown("[Get free key](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.markdown("**Works without key** using rule-based extractor. With key → Gemini JSON.")

samples = {}
for name, fn in [("Sprint Planning", BASE / "samples/sprint_planning.txt"), ("Client Call", BASE / "samples/client_call.txt"), ("College Project", BASE / "samples/college_project.txt")]:
    try: samples[name] = open(fn, encoding="utf-8").read()
    except Exception: samples[name] = ""

c1, c2 = st.columns([2, 1])
with c1:
    choice = st.selectbox("Load sample transcript (or paste/upload your own below)", ["-- Paste / Upload --"] + list(samples.keys()))
with c2:
    meeting_title = st.text_input("Meeting title", "Sprint Planning")

default_text = samples.get(choice, "") if choice in samples else ""
transcript = st.text_area("Paste transcript / notes", value=default_text, height=220, placeholder="Paste meeting transcript, min 50 characters…")
up = st.file_uploader("Or upload .txt", type=["txt"])
if up is not None:
    transcript = up.read().decode("utf-8", errors="ignore")
    st.info(f"Loaded {up.name} ({len(transcript)} chars)")
attendees_raw = st.text_input("Attendees (comma-separated, helps owner detection)", "Priya, Rohan, Sneha, Amit")
attendees = [a.strip().capitalize() for a in attendees_raw.split(",") if a.strip()]

if st.button("✨ Generate Summary + Action Items", use_container_width=True):
    if len(transcript.strip()) < 50:
        st.error("Transcript too short — paste at least 50 characters (edge-case validation). Try a sample above.")
        st.stop()
    used_ai = False
    err = None
    summary, decisions, actions = [], [], []
    if api_key:
        with st.spinner("Asking Gemini…"):
            data, err = gemini_process(api_key, transcript, attendees, meeting_title)
        if data and "actions" in data:
            summary = data.get("summary", [])
            decisions = data.get("decisions", [])
            actions = [{**a, "source": "gemini"} for a in data.get("actions", [])]
            used_ai = True
        else:
            st.warning(f"{err} — falling back to rule-based extractor so demo continues.")
    if not used_ai:
        summary, decisions, actions = rule_based_process(transcript, attendees)
    st.success(f"Done with {'🤖 Gemini AI' if used_ai else '⚙️ Rule-based fallback'} — {len(actions)} actions found.")

    st.subheader("📌 Executive Summary")
    for s in summary: st.write(f"• {s}")
    st.subheader("✅ Key Decisions")
    if decisions:
        for d in decisions: st.write(f"• {d}")
    else: st.info("No explicit decisions detected.")
    st.subheader("📋 Action Items (owner + due)")
    if actions:
        df = pd.DataFrame(actions)[["task", "owner", "due", "priority"]]
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["owner"].value_counts())
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Export task list (CSV)", csv, "action_items.csv", "text/csv")
        md = to_markdown(meeting_title, summary, decisions, df)
        st.download_button("⬇️ Export full notes (Markdown)", md.encode(), "meeting_notes.md", "text/markdown")
        with st.expander("Show Markdown preview"):
            st.markdown(md)
    else:
        st.warning("No action items detected — try longer transcript with 'will / by Friday' phrasing.")
    st.caption("⚠️ AI can misread owners/dates — verify before sharing. This is a pre-final draft, not an official minute.")
