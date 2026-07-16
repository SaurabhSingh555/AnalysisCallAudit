# -*- coding: utf-8 -*-
"""
CallAI Analytics - Streamlit App (SaaS Edition)
================================================
Flow:
  1. Silent login to CRM (fixed credentials, no login screen)
  2. Pick client, date range
  3. Fetch CDR report (auto-filtered by company_id)
  4. REMOVE VDCL AGENT CALLS (abandoned calls) - automatically filtered out
  5. Show ONE filtered data table with duration first, S.No, other columns
  6. Pick call-type and count with client-specific duration configs
  7. Choose sort order
  8. Run VAD (Silero) to get Talk Time / Silence / Dead Air / Longest Silence
  9. Transcribe each call with Groq Whisper → Groq LLM sentiment analysis → add Sentiment column
 10. Download final Excel report with two sheets: Call Report + Agent Analytics
 11. Agent-wise analysis sheet included (updated categories: Short<2min, Medium 2-5min, Large>5min)

  🆕 NEW FEATURES:
 12. Complete Call Analysis (ALL CALLS) with VAD
 13. Agent Performance Dashboard with all agents
 14. Email Management UI - Add/Remove mentors, see who gets emails
 15. Send Performance Report to all mentors (complete agent performance)
 16. Send Defaulter Alert to mentors (only defaulters)
 17. Auto-schedule daily at 11 PM for Weebo and Hari Om
"""

import os
import re
import io
import time
import shutil
import smtplib
import subprocess
import tempfile
import threading
import schedule
from datetime import date, timedelta, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import librosa
from bs4 import BeautifulSoup
import streamlit as st
import torch
import groq

# ============================================================
# PAGE CONFIG + SAAS-STYLE THEME
# ============================================================
st.set_page_config(
    page_title="CallAI · Talk-Time + Sentiment",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Inter, Roboto, Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    .callai-hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .callai-hero h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: white;
    }
    .callai-hero p {
        font-size: 15px;
        margin: 0;
        opacity: 0.9;
    }

    .step-card {
        background: #FFFFFF;
        border: 1px solid #ECECF4;
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(20, 20, 43, 0.04);
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #4F46E5;
        color: white;
        font-weight: 700;
        font-size: 14px;
        margin-right: 10px;
    }
    .step-title {
        font-size: 17px;
        font-weight: 700;
        color: #14142B;
        display: inline-flex;
        align-items: center;
        margin-bottom: 6px;
    }
    .step-subtitle {
        color: #6E7191;
        font-size: 13.5px;
        margin: 0 0 16px 40px;
    }

    .metric-pill {
        background: #F5F4FF;
        border: 1px solid #E4E1FF;
        border-radius: 14px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-pill .value {
        font-size: 24px;
        font-weight: 800;
        color: #4F46E5;
    }
    .metric-pill .label {
        font-size: 12.5px;
        color: #6E7191;
        margin-top: 2px;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        border: none;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
    }
    div.stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
        background: linear-gradient(135deg, #16A34A 0%, #22C55E 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.4rem;
    }

    div[role="radiogroup"] label {
        border: 1px solid #E4E1FF;
        padding: 6px 14px;
        border-radius: 20px;
        margin-right: 6px;
    }

    .status-banner-ok {
        background: #ECFDF5;
        border: 1px solid #6EE7B7;
        color: #065F46;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .status-banner-warning {
        background: #FEF3C7;
        border: 1px solid #FCD34D;
        color: #92400E;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .status-banner-danger {
        background: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    .status-banner-success {
        background: #D1FAE5;
        border: 1px solid #6EE7B7;
        color: #065F46;
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 13.5px;
    }
    
    .agent-card {
        background: #F8F7FF;
        border-left: 4px solid #4F46E5;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .agent-card .agent-name {
        font-weight: 700;
        color: #14142B;
        font-size: 15px;
    }
    .agent-card .agent-stats {
        color: #6E7191;
        font-size: 13px;
        margin-top: 4px;
    }
    .agent-card .highlight {
        color: #4F46E5;
        font-weight: 600;
    }
    .agent-card.warning {
        border-left-color: #EF4444;
        background: #FEF2F2;
    }
    .agent-card.warning .agent-name {
        color: #DC2626;
    }
    .agent-card.success {
        border-left-color: #16A34A;
        background: #F0FDF4;
    }
    .agent-card.success .agent-name {
        color: #16A34A;
    }
    .agent-card.defaulter {
        border-left-color: #DC2626;
        background: #FEE2E2;
        border: 2px solid #DC2626;
    }
    .agent-card.defaulter .agent-name {
        color: #991B1B;
        font-weight: 800;
        font-size: 16px;
    }
    
    .config-box {
        background: #F8F7FF;
        border: 2px solid #E4E1FF;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
    }
    
    .email-chip {
        display: inline-block;
        background: #E5E7EB;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 3px 5px;
        font-size: 13px;
    }
    .email-chip .remove-btn {
        cursor: pointer;
        color: #DC2626;
        margin-left: 6px;
        font-weight: bold;
    }
    
    .email-list-box {
        background: #F8F7FF;
        border: 1px solid #E4E1FF;
        border-radius: 10px;
        padding: 12px 16px;
        min-height: 50px;
        margin-top: 8px;
    }
    
    .email-status-box {
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .email-success {
        background: #D1FAE5;
        border: 1px solid #6EE7B7;
        color: #065F46;
    }
    .email-error {
        background: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
    }
    .email-info {
        background: #EFF6FF;
        border: 1px solid #93C5FD;
        color: #1E40AF;
    }
    
    .schedule-box {
        background: #F0FDF4;
        border: 2px solid #86EFAC;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 10px 0;
    }
    .schedule-box .time {
        font-size: 28px;
        font-weight: 800;
        color: #16A34A;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="callai-hero">
    <h1>📞 CallAI · Talk-Time + Sentiment</h1>
    <p>Pick a client, fetch calls, filter, get Talk-Time / Silence / Dead-Air, and now also analyse sentiment with Groq Whisper + Groq LLM.</p>
</div>
""", unsafe_allow_html=True)

CRM_BASE = "https://crmapi.dialdesk.in"
LOGIN_URL = f"{CRM_BASE}/auth/login"
CDR_URL = f"{CRM_BASE}/report/cdr_report"

# ============================================================
# ⚠️ FIXED CRM CREDENTIALS - Loaded from Streamlit Secrets (with fallback)
# ============================================================
try:
    CRM_EMAIL = st.secrets["CRM_EMAIL"]
    CRM_PASSWORD = st.secrets["CRM_PASSWORD"]
except:
    CRM_EMAIL = "ispark@dialdesk.in"
    CRM_PASSWORD = "1234"

# ============================================================
# GROQ API KEY - load from secrets
# ============================================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = ""

# ============================================================
# ⚠️ SMTP CONFIG - For emails (UPDATED WITH SECRETS)
# ============================================================
def _get_secret(key, default):
    """
    Fetch a value from st.secrets, falling back to `default` if the key is
    missing OR no secrets file exists at all.

    NOTE: this uses bracket-access (st.secrets[key]) inside try/except,
    same pattern as CRM_EMAIL / CRM_PASSWORD / GROQ_API_KEY above.

    This was the actual bug: the previous code used
    `st.secrets.get("SMTP_PASSWORD", "")`, and `.get()` never raises even
    when the key is absent from secrets.toml — it just silently returns the
    literal default passed to `.get()` (an empty string), NOT the working
    hardcoded fallback password that lived in the `except` block below it.
    So whenever a secrets.toml existed (e.g. with CRM_EMAIL/GROQ_API_KEY set
    for deployment) but without SMTP_PASSWORD explicitly added, SMTP_CONFIG
    ended up with an empty password, login failed, and no report/alert
    email ever actually went out to the mentor addresses — with no obvious
    error surfaced beyond "SMTP password not configured".
    """
    try:
        return st.secrets[key]
    except Exception:
        return default

SMTP_CONFIG = {
    "host": "smtp.gmail.com",
    "port": 587,
    "username": _get_secret("SMTP_USERNAME", "singhsolu34907@gmail.com"),
    "password": _get_secret("SMTP_PASSWORD", "lmvdtwcbmcujidld"),
    "from_email": _get_secret("SMTP_FROM_EMAIL", "singhsolu34907@gmail.com"),
    "use_tls": True,
}

# Default mentor emails - stored in session state so user can modify
DEFAULT_MENTOR_EMAILS = [
    "urvi.wadhwa@teammas.in",
]

# Auto-schedule clients - Daily 11 PM for these clients
AUTO_SCHEDULE_CLIENTS = ["Weebo", "Hari Om Pvt Ltd"]

# ============================================================
# ⚠️ CLIENTS - name -> company_id (edit this dict to add/remove clients)
# ============================================================
CLIENTS = {
    "Weebo": "687",
    "Hari Om Pvt Ltd": "689",
    "F1 INFO SOLUTION": "609",
    "Saatvik": "663",
    "Fortum Charge": "395",
    "Alphanso": "629",
}

# ============================================================
# 🆕 CLIENT-SPECIFIC DURATION CONFIGURATIONS
# ============================================================
DEFAULT_DURATION_CONFIG = {
    "short_max": 120,      # seconds (2 minutes)
    "medium_min": 120,     # seconds (2 minutes)
    "medium_max": 300,     # seconds (5 minutes)
    "large_min": 300,      # seconds (5 minutes)
}

CLIENT_DURATION_CONFIGS = {
    "Weebo": {
        "short_max": 90,      # 1.5 minutes
        "medium_min": 90,
        "medium_max": 240,    # 4 minutes
        "large_min": 240,
    },
    "Hari Om Pvt Ltd": {
        "short_max": 150,     # 2.5 minutes
        "medium_min": 150,
        "medium_max": 360,    # 6 minutes
        "large_min": 360,
    },
}

def get_duration_config(client_name):
    config = DEFAULT_DURATION_CONFIG.copy()
    if client_name in CLIENT_DURATION_CONFIGS:
        config.update(CLIENT_DURATION_CONFIGS[client_name])
    session_overrides = st.session_state.get("duration_overrides", {})
    if client_name in session_overrides:
        config.update(session_overrides[client_name])
    return config

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
defaults = {
    "token": None,
    "cdr_df": None,
    "cdr_client": None,
    "final_df": None,
    "agent_analytics_df": None,
    "vdcl_removed": 0,
    "duration_overrides": {},
    "complete_analysis_df": None,
    "agent_silence_by_category_df": None,
    "defaulter_agents_df": None,
    "mentor_emails": DEFAULT_MENTOR_EMAILS.copy(),
    "email_status": None,  # Stores last email send status
    "email_status_message": "",
    "last_email_time": None,
    "scheduler_running": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# CRM FUNCTIONS
# ============================================================
def do_login():
    resp = requests.post(
        LOGIN_URL,
        json={"email": CRM_EMAIL, "password": CRM_PASSWORD},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=30,
        proxies=None,
    )
    resp.raise_for_status()
    data = resp.json()
    token = (
        data.get("token")
        or data.get("access_token")
        or (data.get("data", {}) or {}).get("token")
    )
    if not token:
        raise RuntimeError(f"Login response had no token field: {data}")
    st.session_state["token"] = token
    return token

def get_valid_token():
    if not st.session_state.get("token"):
        with st.spinner("Signing in..."):
            do_login()
    return st.session_state["token"]

def fetch_cdr(payload, retry_on_401=True):
    token = get_valid_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resp = requests.post(CDR_URL, json=payload, headers=headers, timeout=120, proxies=None)
    if resp.status_code == 401 and retry_on_401:
        do_login()
        return fetch_cdr(payload, retry_on_401=False)
    return resp

# ============================================================
# RECORDING DOWNLOAD FUNCTIONS
# ============================================================
def html_recording_to_direct_url(webform_url, retries=3):
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    audio_exts = (".mp3", ".wav", ".m4a", ".mp4")
    for attempt in range(retries):
        try:
            resp = session.get(webform_url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            
            content_type = resp.headers.get('content-type', '').lower()
            if 'audio' in content_type or 'video' in content_type:
                return resp.url
            
            if resp.url.lower().endswith(audio_exts):
                return resp.url
            
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            
            for tag in soup.find_all(["audio", "video"]):
                src = tag.get("src")
                if src and any(ext in src.lower() for ext in audio_exts):
                    return urljoin(webform_url, src)
            for tag in soup.find_all("source"):
                src = tag.get("src")
                if src and any(ext in src.lower() for ext in audio_exts):
                    return urljoin(webform_url, src)
            for div in soup.find_all(attrs={"data-recording": True}):
                for attr in ["data-recording", "data-url", "data-src", "data-file"]:
                    url = div.get(attr)
                    if url:
                        return urljoin(webform_url, url)
            patterns = [
                r'https?://[^\s"\']+\.(?:mp3|wav|m4a)',
                r'//[^\s"\']+\.(?:mp3|wav|m4a)',
                r'/[^\s"\']+\.(?:mp3|wav|m4a)',
            ]
            for pattern in patterns:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    match = m.group()
                    if match.startswith("//"):
                        return "https:" + match
                    if match.startswith("/"):
                        return urljoin(webform_url, match)
                    return match
            js_patterns = [
                r'recordingUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'audioUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'fileUrl\s*[:=]\s*["\']([^"\']+)["\']',
                r'src\s*[:=]\s*["\']([^"\']+\.(?:mp3|wav|m4a))["\']',
            ]
            for pattern in js_patterns:
                m = re.search(pattern, html, re.IGNORECASE)
                if m:
                    return urljoin(webform_url, m.group(1))
            iframe = soup.find("iframe")
            if iframe and iframe.get("src"):
                iframe_src = urljoin(webform_url, iframe.get("src"))
                return html_recording_to_direct_url(iframe_src, retries=retries - 1)
            meta_refresh = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
            if meta_refresh and meta_refresh.get("content"):
                m = re.search(r"url=([^;]+)", meta_refresh.get("content"), re.IGNORECASE)
                if m:
                    return html_recording_to_direct_url(urljoin(webform_url, m.group(1)), retries=retries - 1)
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                if any(ext in href.lower() for ext in audio_exts):
                    return urljoin(webform_url, href)
            return None
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1)
    return None

def resolve_audio_url(recording_url):
    if not isinstance(recording_url, str) or not recording_url.strip():
        return None
    recording_url = recording_url.strip()
    if recording_url.lower().endswith((".mp3", ".wav", ".m4a", ".mp4")):
        return recording_url
    return html_recording_to_direct_url(recording_url)

# ============================================================
# FLEXIBLE COLUMN MAPPING
# ============================================================
COLUMN_CANDIDATES = {
    "date": ["call_date", "CallDate", "Date"],
    "time": ["start_time", "Time", "StartTime"],
    "agent_name": ["full_name", "AgentName", "agent", "Agent Name", "agent_name"],
    "call_from": ["phone_number", "PhoneNumber", "Call From"],
    "recording": ["Recording", "RecordingUrl", "RecordingURL", "recording_url"],
}

def find_column(df, keys):
    lower_map = {c.lower(): c for c in df.columns}
    for key in keys:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None

def parse_duration_series_to_seconds(series):
    s = series.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    needs_time_parse = numeric.isna() & s.str.contains(":", na=False)
    if needs_time_parse.any():
        def to_seconds(val):
            parts = val.split(":")
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                return np.nan
            if len(parts) == 3:
                h, m, sec = parts
                return h * 3600 + m * 60 + sec
            elif len(parts) == 2:
                m, sec = parts
                return m * 60 + sec
            return np.nan
        numeric.loc[needs_time_parse] = s.loc[needs_time_parse].apply(to_seconds)
    return numeric

def resolve_duration_column(df):
    candidates_in_order = [
        ("call_duration", "sec"),
        ("call_duration1", "sec"),
        ("CallDurationSecond", "sec"),
        ("Talkduration", "sec"),
        ("CallDurationMinute", "min"),
    ]
    best_col, best_seconds, best_score = None, None, -1
    for name, unit in candidates_in_order:
        col = find_column(df, [name])
        if not col:
            continue
        seconds = parse_duration_series_to_seconds(df[col])
        if unit == "min":
            seconds = seconds * 60
        non_null = seconds.notna().sum()
        non_zero = (seconds.fillna(0) > 0).sum()
        score = non_zero
        if non_null > 0 and score > best_score:
            best_col, best_seconds, best_score = col, seconds.fillna(0), score
    return best_col, best_seconds

def fmt_hms(total_seconds):
    if total_seconds is None or (isinstance(total_seconds, float) and np.isnan(total_seconds)):
        return "-"
    total_seconds = int(round(total_seconds))
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def filter_out_vdcl_calls(df):
    if df is None or len(df) == 0:
        return df, 0
    
    agent_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'agent name' in col_lower or 'agentname' in col_lower or 'full_name' in col_lower or 'agent' in col_lower:
            agent_col = col
            break
    
    if agent_col is None:
        return df, 0
    
    agent_values = df[agent_col].fillna('').astype(str)
    mask = agent_values.str.contains('VDCL', case=False, na=False, regex=False)
    removed_count = int(mask.sum())
    filtered_df = df[~mask].copy()
    return filtered_df, removed_count

# ============================================================
# AGENT ANALYTICS FUNCTION
# ============================================================
def generate_agent_analytics(df, duration_col='_duration_sec', duration_config=None):
    if df is None or len(df) == 0:
        return None
    
    if duration_config is None:
        duration_config = DEFAULT_DURATION_CONFIG

    agent_col = None
    for col in ['full_name', 'agent', 'Agent Name', 'AgentName', 'agent_name']:
        if col in df.columns:
            agent_col = col
            break
    
    if agent_col is None:
        return None
    
    df_copy = df.copy()
    if duration_col not in df_copy.columns:
        return None
    
    has_xl = duration_config.get('extra_large_enabled', False)

    def categorize_call(duration):
        if duration < duration_config['short_max']:
            return 'Short'
        elif duration <= duration_config['medium_max']:
            return 'Medium'
        elif has_xl and duration > duration_config['extra_large_min']:
            return 'Extra Large'
        else:
            return 'Large'
    
    df_copy['Call_Category'] = df_copy[duration_col].apply(categorize_call)
    
    agent_stats = df_copy.groupby(agent_col).agg({
        duration_col: ['count', 'mean', 'sum'],
        'Call_Category': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    agent_stats.columns = ['Agent', 'Total_Calls', 'Avg_Duration', 'Total_Duration', 'Category_Counts']
    
    def extract_category_counts(category_dict, category):
        return category_dict.get(category, 0)
    
    agent_stats['Short_Calls'] = agent_stats['Category_Counts'].apply(
        lambda x: extract_category_counts(x, 'Short')
    )
    agent_stats['Medium_Calls'] = agent_stats['Category_Counts'].apply(
        lambda x: extract_category_counts(x, 'Medium')
    )
    agent_stats['Large_Calls'] = agent_stats['Category_Counts'].apply(
        lambda x: extract_category_counts(x, 'Large')
    )
    if has_xl:
        agent_stats['Extra_Large_Calls'] = agent_stats['Category_Counts'].apply(
            lambda x: extract_category_counts(x, 'Extra Large')
        )
    
    agent_stats['Short_%'] = (agent_stats['Short_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Medium_%'] = (agent_stats['Medium_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    agent_stats['Large_%'] = (agent_stats['Large_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    if has_xl:
        agent_stats['Extra_Large_%'] = (agent_stats['Extra_Large_Calls'] / agent_stats['Total_Calls'] * 100).round(2)
    
    agent_stats['Avg_Duration_Formatted'] = agent_stats['Avg_Duration'].apply(fmt_hms)
    agent_stats['Total_Duration_Formatted'] = agent_stats['Total_Duration'].apply(fmt_hms)
    agent_stats = agent_stats.drop('Category_Counts', axis=1)
    agent_stats = agent_stats.sort_values('Total_Calls', ascending=False)
    agent_stats['Rank'] = range(1, len(agent_stats) + 1)
    
    return agent_stats

# ============================================================
# GROQ FUNCTIONS
# ============================================================
def load_sentiment_pipeline():
    return "groq" if GROQ_API_KEY else None

def groq_transcribe(audio_file_path, api_key):
    if not api_key:
        return ""
    try:
        client = groq.Groq(api_key=api_key)
        with open(audio_file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en"
            )
        return transcription
    except Exception as e:
        return ""

def analyze_sentiment(text, pipeline):
    if not text or not text.strip():
        return "Neutral"
    if pipeline is None or not GROQ_API_KEY:
        return "Neutral"
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the sentiment of the given call transcript as exactly one "
                        "word: Positive, Negative, or Neutral. Reply with only that one word, "
                        "nothing else."
                    ),
                },
                {"role": "user", "content": text[:3000]},
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = resp.choices[0].message.content.strip()
        label = raw.split()[0].strip(".,!").capitalize() if raw else ""
        if label in ("Positive", "Negative", "Neutral"):
            return label
        return "Neutral"
    except Exception:
        return "Neutral"

# ============================================================
# 🆕 VAD FUNCTIONS - COMPLETE CALL ANALYSIS
# ============================================================

@st.cache_resource(show_spinner="🔄 Loading voice-detection model (first run only)...")
def load_vad_model():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("⏳ Setting up model directory...")
    progress_bar.progress(20)
    
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
    except Exception:
        hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub")
        os.makedirs(hub_dir, exist_ok=True)
    
    torch.hub.set_dir(hub_dir)
    
    status_text.text("⏳ Downloading Silero VAD model...")
    progress_bar.progress(50)
    
    try:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
        )
    except Exception:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", force_reload=False
        )
    
    progress_bar.progress(100)
    status_text.text("✅ VAD model loaded successfully!")
    time.sleep(2)
    progress_bar.empty()
    status_text.empty()
    
    return model, utils

def robust_normalize(audio):
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms > 1e-4:
        target_rms = 0.1
        gain = target_rms / rms
        gain = min(gain, 20.0)
        audio = audio * gain
    return np.clip(audio, -1.0, 1.0)

def load_channel_16k(data, sr, channel_idx=None):
    if data.ndim > 1:
        chan = data[:, channel_idx] if channel_idx is not None else np.mean(data, axis=1)
    else:
        chan = data
    chan = robust_normalize(chan.astype(np.float32))
    if sr != 16000:
        chan = librosa.resample(chan, orig_sr=sr, target_sr=16000)
    return torch.from_numpy(chan).float()

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

def compute_metrics(intervals, total_duration, dead_air_secs):
    if not intervals:
        return {
            "talk_time": 0.0,
            "silence_time": round(total_duration, 2),
            "dead_air": round(total_duration, 2) if total_duration > dead_air_secs else 0.0,
            "longest_silence": round(total_duration, 2),
        }
    speech_time, longest_silence, dead_air, prev_end = 0.0, 0.0, 0.0, 0.0
    for s, e in intervals:
        speech_time += (e - s)
        silence = max(0.0, s - prev_end)
        longest_silence = max(longest_silence, silence)
        if silence > dead_air_secs:
            dead_air += silence
        prev_end = e
    ending_silence = max(0.0, total_duration - prev_end)
    longest_silence = max(longest_silence, ending_silence)
    if ending_silence > dead_air_secs:
        dead_air += ending_silence
    silence_time = max(0.0, total_duration - speech_time)
    return {
        "talk_time": round(speech_time, 2),
        "silence_time": round(silence_time, 2),
        "dead_air": round(dead_air, 2),
        "longest_silence": round(longest_silence, 2),
    }

def process_recording(rec_url, model, get_speech_timestamps, vad_threshold, dead_air_secs, tmpdir, tag):
    metrics = {"talk_time": None, "silence_time": None, "dead_air": None, "longest_silence": None, "duration": None}
    debug_status = "OK"
    actual_mp3 = None
    mp3_path = os.path.join(tmpdir, f"{tag}.mp3")
    wav_path = os.path.join(tmpdir, f"{tag}.wav")
    
    try:
        if not rec_url:
            return metrics, "No recording URL in this row", None
        
        actual_mp3 = resolve_audio_url(rec_url)
        if not actual_mp3:
            return metrics, "Could not resolve a direct audio URL from the recording link", None
        
        r = requests.get(actual_mp3, timeout=60, stream=True)
        if r.status_code != 200:
            return metrics, f"Download failed: HTTP {r.status_code}", actual_mp3
        with open(mp3_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            return metrics, "Downloaded file is empty", actual_mp3
        
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if ff.returncode != 0 or not os.path.exists(wav_path):
            err_tail = ff.stderr.decode(errors="ignore")[-300:]
            return metrics, f"FFmpeg conversion failed: {err_tail.strip()}", actual_mp3
        
        data, sr = sf.read(wav_path)
        total_duration = len(data) / sr
        is_stereo = data.ndim > 1 and data.shape[1] > 1
        
        vad_kwargs = dict(
            sampling_rate=16000,
            threshold=vad_threshold,
            min_speech_duration_ms=100,
            min_silence_duration_ms=200,
            speech_pad_ms=300,
            window_size_samples=512,
        )
        
        if is_stereo:
            all_intervals = []
            for ch in range(data.shape[1]):
                tensor = load_channel_16k(data, sr, channel_idx=ch)
                ts = get_speech_timestamps(tensor, model, **vad_kwargs)
                all_intervals.extend([(s["start"] / 16000, s["end"] / 16000) for s in ts])
            merged = merge_intervals(all_intervals)
        else:
            tensor = load_channel_16k(data, sr)
            ts = get_speech_timestamps(tensor, model, **vad_kwargs)
            merged = merge_intervals([(s["start"] / 16000, s["end"] / 16000) for s in ts])
        
        metrics = compute_metrics(merged, total_duration, dead_air_secs)
        metrics["duration"] = round(total_duration, 2)
        return metrics, "OK", actual_mp3
    except requests.exceptions.RequestException as e:
        return metrics, f"Network/download error: {str(e)[:100]}", actual_mp3
    except Exception as e:
        return metrics, f"Processing error: {str(e)[:100]}", actual_mp3
    finally:
        if os.path.exists(mp3_path):
            try: os.remove(mp3_path)
            except Exception: pass
        if os.path.exists(wav_path):
            try: os.remove(wav_path)
            except Exception: pass

# ============================================================
# 🆕 COMPLETE CALL ANALYSIS FUNCTIONS
# ============================================================

def detect_defaulters_simple(silence_df, duration_config, silence_threshold=30, min_calls=3):
    """
    Simple defaulter detection based on:
    - Overall avg silence %
    - Short call avg silence %
    """
    if silence_df is None or len(silence_df) == 0:
        return None
    
    short_max = duration_config.get('short_max', 120)
    medium_max = duration_config.get('medium_max', 300)
    
    def categorize_by_duration(duration):
        if duration is None or pd.isna(duration):
            return 'Unknown'
        if duration < short_max:
            return 'Short'
        elif duration <= medium_max:
            return 'Medium'
        else:
            return 'Large'
    
    silence_df_copy = silence_df.copy()
    silence_df_copy['Duration (sec)'] = pd.to_numeric(silence_df_copy['Duration (sec)'], errors='coerce')
    silence_df_copy['Category'] = silence_df_copy['Duration (sec)'].apply(categorize_by_duration)
    silence_df_copy = silence_df_copy[silence_df_copy['Category'] != 'Unknown']
    
    if len(silence_df_copy) == 0:
        return None
    
    agent_data = []
    
    for agent in silence_df_copy['Agent Name'].unique():
        agent_calls = silence_df_copy[silence_df_copy['Agent Name'] == agent]
        total_calls = len(agent_calls)
        
        if total_calls < min_calls:
            continue
        
        overall_avg_silence = agent_calls['Silence %'].mean()
        
        short_calls = agent_calls[agent_calls['Category'] == 'Short']
        medium_calls = agent_calls[agent_calls['Category'] == 'Medium']
        large_calls = agent_calls[agent_calls['Category'] == 'Large']
        
        short_avg_silence = short_calls['Silence %'].mean() if len(short_calls) > 0 else 0
        medium_avg_silence = medium_calls['Silence %'].mean() if len(medium_calls) > 0 else 0
        large_avg_silence = large_calls['Silence %'].mean() if len(large_calls) > 0 else 0
        
        short_count = len(short_calls)
        medium_count = len(medium_calls)
        large_count = len(large_calls)
        
        is_defaulter = (overall_avg_silence > silence_threshold) or (short_avg_silence > silence_threshold)
        
        if is_defaulter:
            agent_data.append({
                'Agent': agent,
                'Total_Calls': total_calls,
                'Short_Calls': short_count,
                'Short_Silence_%': round(short_avg_silence, 1),
                'Medium_Calls': medium_count,
                'Medium_Silence_%': round(medium_avg_silence, 1),
                'Large_Calls': large_count,
                'Large_Silence_%': round(large_avg_silence, 1),
                'Overall_Silence_%': round(overall_avg_silence, 1),
            })
    
    if len(agent_data) == 0:
        return None
    
    df_defaulters = pd.DataFrame(agent_data)
    df_defaulters = df_defaulters.sort_values('Overall_Silence_%', ascending=False)
    df_defaulters['Rank'] = range(1, len(df_defaulters) + 1)
    
    return df_defaulters

def analyze_agent_silence_by_category(silence_df, duration_config):
    if silence_df is None or len(silence_df) == 0:
        return None
    
    short_max = duration_config.get('short_max', 120)
    medium_max = duration_config.get('medium_max', 300)
    
    def categorize_by_duration(duration):
        if duration is None or pd.isna(duration):
            return 'Unknown'
        if duration < short_max:
            return 'Short'
        elif duration <= medium_max:
            return 'Medium'
        else:
            return 'Large'
    
    silence_df_copy = silence_df.copy()
    silence_df_copy['Duration (sec)'] = pd.to_numeric(silence_df_copy['Duration (sec)'], errors='coerce')
    silence_df_copy['Category'] = silence_df_copy['Duration (sec)'].apply(categorize_by_duration)
    
    agent_category_silence = silence_df_copy.groupby(['Agent Name', 'Category']).agg({
        'Silence %': ['mean', 'count'],
    }).reset_index()
    
    agent_category_silence.columns = ['Agent', 'Category', 'Avg_Silence_%', 'Call_Count']
    
    pivot_df = agent_category_silence.pivot_table(
        index='Agent',
        columns='Category',
        values=['Avg_Silence_%', 'Call_Count'],
        fill_value=0
    ).reset_index()
    
    pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns.values]
    pivot_df = pivot_df.rename(columns={'Agent_': 'Agent'})
    
    for cat in ['Short', 'Medium', 'Large']:
        if f'Avg_Silence_%_{cat}' not in pivot_df.columns:
            pivot_df[f'Avg_Silence_%_{cat}'] = 0
        if f'Call_Count_{cat}' not in pivot_df.columns:
            pivot_df[f'Call_Count_{cat}'] = 0
    
    agent_overall = silence_df_copy.groupby('Agent Name').agg({
        'Silence %': 'mean',
        'Silence Time (sec)': 'sum',
        'Duration (sec)': 'sum'
    }).reset_index()
    agent_overall.columns = ['Agent', 'Overall_Avg_Silence_%', 'Total_Silence_Time', 'Total_Duration']
    
    pivot_df = pivot_df.merge(agent_overall, on='Agent', how='left')
    
    pivot_df['Total_Calls'] = pivot_df['Call_Count_Short'] + pivot_df['Call_Count_Medium'] + pivot_df['Call_Count_Large']
    pivot_df = pivot_df.sort_values('Overall_Avg_Silence_%', ascending=False).reset_index(drop=True)
    pivot_df['Rank'] = range(1, len(pivot_df) + 1)
    
    return pivot_df

def run_complete_call_analysis(all_calls_df, col_recording, col_agent, col_date, col_time, col_phone,
                               vad_threshold, dead_air_secs, duration_config, 
                               silence_threshold=30, min_calls=3):
    """
    Runs VAD analysis on ALL calls and returns silence data, performance dashboard, and defaulter detection.
    """
    model, utils = load_vad_model()
    get_speech_timestamps = utils[0]
    
    progress = st.progress(0)
    status = st.empty()
    total = len(all_calls_df)
    silence_rows = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (_, row) in enumerate(all_calls_df.iterrows()):
            status.text(f"Analyzing call {i+1}/{total}...")
            metrics, debug_status, actual_mp3 = process_recording(
                row.get(col_recording), model, get_speech_timestamps,
                vad_threshold, dead_air_secs, tmpdir, f"complete_{i}",
            )
            dur = metrics.get("duration")
            sil = metrics.get("silence_time")
            silence_pct = round((sil / dur * 100), 1) if dur and dur > 0 and sil is not None else None
            
            date_val = row.get(col_date) if col_date else None
            time_val = row.get(col_time) if col_time else None
            
            if date_val and pd.notna(date_val):
                try:
                    date_val = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                except:
                    pass
            
            silence_rows.append({
                "Date": date_val,
                "Time": time_val,
                "Agent Name": row.get(col_agent) if col_agent else None,
                "Call From": row.get(col_phone) if col_phone else None,
                "Duration (sec)": dur,
                "Silence Time (sec)": sil,
                "Silence %": silence_pct,
                "Longest Silence (sec)": metrics.get("longest_silence"),
                "Duration_CRM": row.get("_duration_sec"),
                "Status": debug_status,
                "Actual MP3": actual_mp3,
            })
            progress.progress((i + 1) / total)
    
    status.text("✅ Complete call analysis done!")
    silence_df = pd.DataFrame(silence_rows)
    
    perf_config = duration_config.copy()
    perf_df = analyze_agent_silence_by_category(silence_df, perf_config)
    
    defaulters_df = detect_defaulters_simple(
        silence_df, 
        duration_config, 
        silence_threshold=silence_threshold,
        min_calls=min_calls
    )
    
    return silence_df, perf_df, defaulters_df

# ============================================================
# 🆕 EMAIL FUNCTIONS WITH FIXES
# ============================================================

def test_smtp_connection():
    """Test SMTP connection before sending"""
    try:
        if not SMTP_CONFIG.get("password"):
            return False, "SMTP password not configured. Please add SMTP_PASSWORD to secrets."
        
        print(f"Testing SMTP connection to {SMTP_CONFIG['host']}:{SMTP_CONFIG['port']}")
        
        with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"], timeout=30) as server:
            server.ehlo()
            if SMTP_CONFIG.get("use_tls", True):
                print("Starting TLS...")
                server.starttls()
                server.ehlo()
            
            print(f"Logging in as {SMTP_CONFIG['username']}...")
            server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
            print("✅ SMTP connection successful!")
            
        return True, "SMTP connection successful"
    except Exception as e:
        error_msg = str(e)
        print(f"❌ SMTP Error: {error_msg}")
        
        if "Authentication failed" in error_msg:
            return False, "Authentication failed. Check your email/password or generate an App Password."
        elif "Connection refused" in error_msg:
            return False, "Connection refused. Check host and port."
        elif "timed out" in error_msg:
            return False, "Connection timed out. Check your internet."
        else:
            return False, f"SMTP Error: {error_msg}"

def send_performance_report_email(perf_df, silence_df, client_name, mentor_emails):
    """
    Send complete performance report to all mentors with better error handling
    """
    if perf_df is None or len(perf_df) == 0:
        return False, "No performance data available"
    
    if not mentor_emails:
        return False, "No mentor emails configured!"
    
    # Clean emails
    mentor_emails = [e.strip() for e in mentor_emails if e and '@' in e]
    if not mentor_emails:
        return False, "No valid mentor emails found!"
    
    # Test SMTP connection first
    smtp_ok, smtp_msg = test_smtp_connection()
    if not smtp_ok:
        return False, f"SMTP Error: {smtp_msg}"
    
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_CONFIG.get("from_email") or SMTP_CONFIG["username"]
        msg["To"] = ", ".join(mentor_emails)
        msg["Subject"] = f"📊 Agent Performance Report — {client_name} ({date.today().strftime('%d/%m/%Y')})"
        
        # Build agent performance table
        agent_rows = ""
        for _, row in perf_df.iterrows():
            agent = row.get('Agent', 'Unknown')
            total = row.get('Total_Calls', 0)
            short = row.get('Avg_Silence_%_Short', 0)
            medium = row.get('Avg_Silence_%_Medium', 0)
            large = row.get('Avg_Silence_%_Large', 0)
            overall = row.get('Overall_Avg_Silence_%', 0)
            
            if overall > 40:
                color = '#DC2626'
            elif overall > 30:
                color = '#D97706'
            else:
                color = '#16A34A'
            
            agent_rows += f"""
            <tr>
                <td style="padding:8px;font-weight:bold;">{agent}</td>
                <td style="padding:8px;text-align:center;">{total}</td>
                <td style="padding:8px;text-align:center;">{short:.1f}%</td>
                <td style="padding:8px;text-align:center;">{medium:.1f}%</td>
                <td style="padding:8px;text-align:center;">{large:.1f}%</td>
                <td style="padding:8px;text-align:center;font-weight:bold;color:{color};">{overall:.1f}%</td>
            </tr>
            """
        
        # Summary stats
        avg_overall = perf_df['Overall_Avg_Silence_%'].mean() if 'Overall_Avg_Silence_%' in perf_df else 0
        best_agent = perf_df.iloc[0].get('Agent', 'N/A') if len(perf_df) > 0 else "N/A"
        worst_agent = perf_df.iloc[-1].get('Agent', 'N/A') if len(perf_df) > 0 else "N/A"
        total_agents = len(perf_df)
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: #4F46E5; padding: 20px; border-radius: 8px; color: white; margin-bottom: 20px; }}
                .summary {{ background: #F3F4F6; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #1F2937; color: white; padding: 10px; text-align: center; }}
                td {{ padding: 8px; text-align: center; border-bottom: 1px solid #E5E7EB; }}
                .footer {{ margin-top: 20px; color: #6B7280; font-size: 12px; text-align: center; }}
                .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
                .badge-good {{ background: #D1FAE5; color: #065F46; }}
                .badge-warning {{ background: #FEF3C7; color: #92400E; }}
                .badge-danger {{ background: #FEE2E2; color: #991B1B; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📊 Agent Performance Report</h2>
                <p><strong>Client:</strong> {client_name} | <strong>Date:</strong> {date.today().strftime('%d/%m/%Y')}</p>
                <p>Total Agents: {total_agents} | Avg Silence: {avg_overall:.1f}%</p>
            </div>
            
            <div class="summary">
                <table style="width:100%;border:none;">
                    <tr>
                        <td style="border:none;text-align:center;"><strong>🏆 Best Performer</strong><br>{best_agent}</td>
                        <td style="border:none;text-align:center;"><strong>📉 Needs Improvement</strong><br>{worst_agent}</td>
                        <td style="border:none;text-align:center;"><strong>📊 Avg Silence</strong><br>{avg_overall:.1f}%</td>
                        <td style="border:none;text-align:center;"><strong>👥 Total Agents</strong><br>{total_agents}</td>
                    </tr>
                </table>
            </div>
            
            <h3>📋 Agent Performance Table</h3>
            <p><span class="badge badge-good">✅ Good (&lt;30%)</span> 
               <span class="badge badge-warning">⚠️ Medium (30-40%)</span> 
               <span class="badge badge-danger">🔴 High (&gt;40%)</span></p>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Total Calls</th>
                        <th>Short %</th>
                        <th>Medium %</th>
                        <th>Large %</th>
                        <th>Overall %</th>
                    </tr>
                </thead>
                <tbody>
                    {agent_rows}
                </tbody>
            </table>
            
            <div class="footer">
                <p>— CallAI Analytics | Automated Performance Report</p>
                <p>This report shows silence percentage by call category for all agents.</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))
        
        # Attach Excel report
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                perf_df.to_excel(writer, index=False, sheet_name="Performance Report")
                if silence_df is not None and len(silence_df) > 0:
                    silence_df.to_excel(writer, index=False, sheet_name="All Calls Data")
            buf.seek(0)
            
            excel_attachment = MIMEApplication(buf.read(), _subtype="xlsx")
            excel_attachment.add_header('Content-Disposition', 'attachment', filename=f'Performance_Report_{client_name}_{date.today().strftime("%Y%m%d")}.xlsx')
            msg.attach(excel_attachment)
        except Exception as e:
            print(f"Excel attachment error: {e}")
            # Continue without attachment if error
        
        # Send email with better error handling
        try:
            with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"], timeout=30) as server:
                server.ehlo()
                if SMTP_CONFIG.get("use_tls", True):
                    server.starttls()
                    server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(msg["From"], mentor_emails, msg.as_string())
            
            return True, f"Performance report sent to {len(mentor_emails)} mentors!"
            
        except smtplib.SMTPAuthenticationError as e:
            return False, f"Authentication failed: {str(e)}. Please check your email/password."
        except smtplib.SMTPServerDisconnected as e:
            return False, f"Server disconnected: {str(e)}"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
    
    except Exception as e:
        return False, f"Error preparing email: {str(e)}"

def send_defaulter_alert_email(defaulters_df, silence_df, client_name, mentor_emails):
    """
    Send defaulter alert email to mentors with better error handling
    """
    if defaulters_df is None or len(defaulters_df) == 0:
        return False, "No defaulters to report"
    
    if not mentor_emails:
        return False, "No mentor emails configured!"
    
    # Clean emails
    mentor_emails = [e.strip() for e in mentor_emails if e and '@' in e]
    if not mentor_emails:
        return False, "No valid mentor emails found!"
    
    # Test SMTP connection first
    smtp_ok, smtp_msg = test_smtp_connection()
    if not smtp_ok:
        return False, f"SMTP Error: {smtp_msg}"
    
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_CONFIG.get("from_email") or SMTP_CONFIG["username"]
        msg["To"] = ", ".join(mentor_emails)
        msg["Subject"] = f"🚨 Defaulter Alert — {len(defaulters_df)} agent(s) flagged ({client_name})"
        
        # Build agent table
        agent_rows = ""
        for _, row in defaulters_df.iterrows():
            agent = row.get('Agent', 'Unknown')
            total = row.get('Total_Calls', 0)
            short_c = row.get('Short_Calls', 0)
            short_s = row.get('Short_Silence_%', 0)
            med_c = row.get('Medium_Calls', 0)
            med_s = row.get('Medium_Silence_%', 0)
            large_c = row.get('Large_Calls', 0)
            large_s = row.get('Large_Silence_%', 0)
            overall = row.get('Overall_Silence_%', 0)
            
            if overall > 40:
                color = '#DC2626'
                badge = '🔴 HIGH'
            elif overall > 30:
                color = '#D97706'
                badge = '🟠 MEDIUM'
            else:
                color = '#2563EB'
                badge = '🔵 LOW'
            
            agent_rows += f"""
            <tr>
                <td style="padding:10px;font-weight:bold;color:{color};">{agent}</td>
                <td style="padding:10px;text-align:center;">{total}</td>
                <td style="padding:10px;text-align:center;">{short_c}<br><small>{short_s}%</small></td>
                <td style="padding:10px;text-align:center;">{med_c}<br><small>{med_s}%</small></td>
                <td style="padding:10px;text-align:center;">{large_c}<br><small>{large_s}%</small></td>
                <td style="padding:10px;text-align:center;font-weight:bold;color:{color};">{overall}%</td>
                <td style="padding:10px;text-align:center;"><span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:12px;">{badge}</span></td>
            </tr>
            """
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: #FEE2E2; padding: 20px; border-radius: 8px; border-left: 6px solid #DC2626; margin-bottom: 20px; }}
                .summary {{ background: #F3F4F6; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #1F2937; color: white; padding: 10px; text-align: center; }}
                td {{ padding: 8px; text-align: center; border-bottom: 1px solid #E5E7EB; }}
                .footer {{ margin-top: 20px; color: #6B7280; font-size: 12px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 Defaulter Agents Detected</h2>
                <p><strong>Client:</strong> {client_name} | <strong>Date:</strong> {date.today().strftime('%d/%m/%Y')}</p>
                <p><strong>Total Defaulters:</strong> {len(defaulters_df)}</p>
            </div>
            
            <div class="summary">
                <p><strong>📋 What does this mean?</strong></p>
                <p>These agents have <strong>Overall Silence &gt; 30%</strong> OR <strong>Short Call Silence &gt; 30%</strong>.</p>
                <p><span style="color:#DC2626;">🔴 High</span> = &gt;40% | 
                   <span style="color:#D97706;">🟠 Medium</span> = 30-40% | 
                   <span style="color:#2563EB;">🔵 Low</span> = &lt;30%</p>
            </div>
            
            <h3>📊 Defaulter Agents</h3>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Total Calls</th>
                        <th>Short Calls<br><small style="font-weight:normal;">(Avg Silence)</small></th>
                        <th>Medium Calls<br><small style="font-weight:normal;">(Avg Silence)</small></th>
                        <th>Large Calls<br><small style="font-weight:normal;">(Avg Silence)</small></th>
                        <th>Overall<br>Silence %</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
                    {agent_rows}
                </tbody>
            </table>
            
            <div class="footer">
                <p>— CallAI Analytics | Automated Defaulter Detection</p>
                <p>Please review these agents and take appropriate action.</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))
        
        # Attach Excel
        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                defaulters_df.to_excel(writer, index=False, sheet_name="Defaulters")
                if silence_df is not None and len(silence_df) > 0:
                    silence_df.to_excel(writer, index=False, sheet_name="All Calls Data")
            buf.seek(0)
            
            excel_attachment = MIMEApplication(buf.read(), _subtype="xlsx")
            excel_attachment.add_header('Content-Disposition', 'attachment', filename=f'Defaulter_Alert_{client_name}_{date.today().strftime("%Y%m%d")}.xlsx')
            msg.attach(excel_attachment)
        except Exception as e:
            print(f"Excel attachment error: {e}")
            # Continue without attachment if error
        
        # Send email with better error handling
        try:
            with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"], timeout=30) as server:
                server.ehlo()
                if SMTP_CONFIG.get("use_tls", True):
                    server.starttls()
                    server.ehlo()
                server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                server.sendmail(msg["From"], mentor_emails, msg.as_string())
            
            return True, f"Defaulter alert sent to {len(mentor_emails)} mentors!"
            
        except smtplib.SMTPAuthenticationError as e:
            return False, f"Authentication failed: {str(e)}. Please check your email/password."
        except smtplib.SMTPServerDisconnected as e:
            return False, f"Server disconnected: {str(e)}"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
    
    except Exception as e:
        return False, f"Error preparing email: {str(e)}"

# ============================================================
# 🆕 AUTO-SCHEDULER FUNCTION
# ============================================================

def run_scheduled_job():
    """Run the scheduled job for Weebo and Hari Om at 11 PM"""
    try:
        print(f"[{datetime.now()}] Starting scheduled job...")
        
        # Login to CRM
        token = do_login()
        
        # Get today's date
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        # Process each auto-schedule client
        for client_name in AUTO_SCHEDULE_CLIENTS:
            if client_name not in CLIENTS:
                continue
                
            company_id = CLIENTS[client_name]
            print(f"[{datetime.now()}] Processing {client_name}...")
            
            # Fetch CDR for yesterday
            payload = {
                "from_date": yesterday.strftime("%Y-%m-%d"),
                "to_date": yesterday.strftime("%Y-%m-%d"),
                "company_id": str(company_id),
            }
            
            resp = fetch_cdr(payload)
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(records, dict):
                    for v in records.values():
                        if isinstance(v, list):
                            records = v
                            break
                cdr_df = pd.DataFrame(records)
                
                # Filter VDCL calls
                cdr_df, _ = filter_out_vdcl_calls(cdr_df)
                
                # Add duration column
                _, duration_seconds = resolve_duration_column(cdr_df)
                cdr_df["_duration_sec"] = duration_seconds
                
                if len(cdr_df) > 0:
                    # Find columns
                    col_recording = find_column(cdr_df, COLUMN_CANDIDATES["recording"])
                    col_agent = find_column(cdr_df, COLUMN_CANDIDATES["agent_name"])
                    col_date = find_column(cdr_df, COLUMN_CANDIDATES["date"])
                    col_time = find_column(cdr_df, COLUMN_CANDIDATES["time"])
                    col_phone = find_column(cdr_df, COLUMN_CANDIDATES["call_from"])
                    
                    if col_recording:
                        # Run complete analysis
                        duration_config = get_duration_config(client_name)
                        silence_df, perf_df, defaulters_df = run_complete_call_analysis(
                            cdr_df, col_recording, col_agent, col_date, col_time, col_phone,
                            vad_threshold=0.3, dead_air_secs=5, duration_config=duration_config,
                            silence_threshold=30, min_calls=3
                        )
                        
                        # Get mentor emails
                        mentor_emails = st.session_state.get("mentor_emails", DEFAULT_MENTOR_EMAILS)
                        
                        # Send performance report
                        if perf_df is not None and len(perf_df) > 0:
                            success, msg = send_performance_report_email(
                                perf_df, silence_df, client_name, mentor_emails
                            )
                            if success:
                                print(f"[{datetime.now()}] Performance report sent for {client_name}")
                            else:
                                print(f"[{datetime.now()}] Failed to send performance report for {client_name}: {msg}")
                        
                        # Send defaulter alert if any
                        if defaulters_df is not None and len(defaulters_df) > 0:
                            success, msg = send_defaulter_alert_email(
                                defaulters_df, silence_df, client_name, mentor_emails
                            )
                            if success:
                                print(f"[{datetime.now()}] Defaulter alert sent for {client_name}")
                            else:
                                print(f"[{datetime.now()}] Failed to send defaulter alert for {client_name}: {msg}")
            
        print(f"[{datetime.now()}] Scheduled job completed.")
        
    except Exception as e:
        print(f"[{datetime.now()}] Scheduled job error: {str(e)}")

def start_scheduler():
    """Start the background scheduler thread"""
    if st.session_state.get("scheduler_running", False):
        return
    
    def scheduler_loop():
        # Schedule at 23:00 (11 PM) every day
        schedule.every().day.at("23:00").do(run_scheduled_job)
        
        # Run immediately once for testing (uncomment for testing)
        # schedule.every(1).minutes.do(run_scheduled_job)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Start scheduler in background thread
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    st.session_state["scheduler_running"] = True
    print(f"[{datetime.now()}] Scheduler started for 11 PM daily")

# ============================================================
# TEST EMAIL FUNCTION - Add this to debug
# ============================================================

def test_email_sending():
    """Debug function to test email sending"""
    st.markdown("### 📧 Test Email Sending")
    st.markdown("Use this to test your SMTP configuration before sending real reports.")
    
    test_email = st.text_input("Test Email Address", 
                               value=st.session_state.mentor_emails[0] if st.session_state.mentor_emails else "",
                               placeholder="Enter email to test")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔬 Test SMTP Connection", use_container_width=True):
            smtp_ok, smtp_msg = test_smtp_connection()
            if smtp_ok:
                st.markdown(f"""
                <div class="email-status-box email-success">
                    ✅ {smtp_msg}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="email-status-box email-error">
                    ❌ {smtp_msg}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📧 Send Test Email", use_container_width=True, disabled=not test_email):
            if test_email and '@' in test_email:
                try:
                    # Send test email
                    msg = MIMEMultipart()
                    msg["From"] = SMTP_CONFIG.get("from_email") or SMTP_CONFIG["username"]
                    msg["To"] = test_email
                    msg["Subject"] = "🧪 Test Email from CallAI Analytics"
                    
                    body = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif;">
                        <div style="background: #4F46E5; padding: 20px; border-radius: 8px; color: white;">
                            <h2>✅ SMTP Test Successful!</h2>
                            <p>This is a test email from CallAI Analytics.</p>
                            <p><strong>Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            <p><strong>Client:</strong> Test</p>
                        </div>
                        <div style="margin-top: 20px; color: #6B7280; font-size: 12px;">
                            <p>— CallAI Analytics | Automated Testing</p>
                            <p>If you received this email, your SMTP configuration is working correctly!</p>
                        </div>
                    </body>
                    </html>
                    """
                    
                    msg.attach(MIMEText(body, "html"))
                    
                    with smtplib.SMTP(SMTP_CONFIG["host"], SMTP_CONFIG["port"], timeout=30) as server:
                        server.ehlo()
                        if SMTP_CONFIG.get("use_tls", True):
                            server.starttls()
                            server.ehlo()
                        server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
                        server.sendmail(msg["From"], [test_email], msg.as_string())
                    
                    st.markdown(f"""
                    <div class="email-status-box email-success">
                        ✅ Test email sent successfully to <strong>{test_email}</strong>!
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="email-status-box email-error">
                        ❌ Test failed: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a valid email address")

# ============================================================
# MAIN APP STARTS HERE
# ============================================================

# Start scheduler on app load
if not st.session_state.get("scheduler_running", False):
    start_scheduler()

# ============================================================
# STEP 1 — CLIENT + DATE RANGE + FETCH
# ============================================================
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown('<div class="step-title"><span class="step-badge">1</span>Choose Client & Date Range</div>', unsafe_allow_html=True)
st.markdown('<p class="step-subtitle">Only calls belonging to the selected client will be fetched.</p>', unsafe_allow_html=True)

c1, c2 = st.columns([1.2, 1.8])
with c1:
    client_name = st.selectbox("Client", options=sorted(CLIENTS.keys()))
    company_id = CLIENTS[client_name]
with c2:
    dc1, dc2 = st.columns(2)
    with dc1:
        from_date = st.date_input("Start Date", value=date.today() - timedelta(days=1), max_value=date.today())
    with dc2:
        to_date = st.date_input("End Date", value=date.today(), max_value=date.today())
    if from_date > to_date:
        st.error("End Date must be on or after Start Date.")

if st.session_state.get("cdr_client") is not None and st.session_state["cdr_client"] != client_name:
    st.session_state["cdr_df"] = None
    st.session_state["cdr_client"] = None
    st.session_state["final_df"] = None
    st.session_state["agent_analytics_df"] = None
    st.session_state["complete_analysis_df"] = None
    st.session_state["agent_silence_by_category_df"] = None
    st.session_state["defaulter_agents_df"] = None

fetch_clicked = st.button("📥  Fetch Calls", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FETCH CDR REPORT
# ============================================================
if fetch_clicked:
    if from_date > to_date:
        st.error("Please correct the date range before fetching.")
    else:
        try:
            payload = {
                "from_date": from_date.strftime("%Y-%m-%d"),
                "to_date": to_date.strftime("%Y-%m-%d"),
                "company_id": str(company_id),
            }
            with st.spinner(f"Fetching calls for {client_name}..."):
                resp = fetch_cdr(payload)
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(records, dict):
                    for v in records.values():
                        if isinstance(v, list):
                            records = v
                            break
                cdr_df = pd.DataFrame(records)
                
                cdr_df, removed_count = filter_out_vdcl_calls(cdr_df)
                st.session_state["vdcl_removed"] = removed_count
                
                dur_source_col, duration_seconds = resolve_duration_column(cdr_df)
                cdr_df["_duration_sec"] = duration_seconds

                st.session_state["cdr_df"] = cdr_df
                st.session_state["cdr_client"] = client_name
                st.session_state["final_df"] = None
                st.session_state["agent_analytics_df"] = None
                st.session_state["complete_analysis_df"] = None
                st.session_state["agent_silence_by_category_df"] = None
                st.session_state["defaulter_agents_df"] = None
                
                if len(cdr_df) == 0:
                    st.warning(f"No valid calls found for **{client_name}** in this date range.")
                else:
                    st.markdown(
                        f'<span class="status-banner-ok">✅ Fetched {len(cdr_df)} valid calls for {client_name}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.error(f"Fetch failed: HTTP {resp.status_code} — {resp.text[:300]}")
        except Exception as e:
            st.error(f"Fetch error: {e}")

# ============================================================
# STEP 2 — FILTER & DISPLAY
# ============================================================
have_data = (
    st.session_state["cdr_df"] is not None
    and len(st.session_state["cdr_df"]) > 0
    and st.session_state.get("cdr_client") == client_name
)

if have_data:
    cdr_df = st.session_state["cdr_df"].copy()
    col_date = find_column(cdr_df, COLUMN_CANDIDATES["date"])
    col_time = find_column(cdr_df, COLUMN_CANDIDATES["time"])
    col_agent = find_column(cdr_df, COLUMN_CANDIDATES["agent_name"])
    col_phone = find_column(cdr_df, COLUMN_CANDIDATES["call_from"])
    col_recording = find_column(cdr_df, COLUMN_CANDIDATES["recording"])
    dur_source_col, duration_seconds = resolve_duration_column(cdr_df)

    cdr_df, _ = filter_out_vdcl_calls(cdr_df)

    if dur_source_col is None:
        st.error("Could not find a usable call-duration column.")
        st.stop()

    cdr_df["_duration_sec"] = duration_seconds

    # ---- Step 2 card ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">2</span>Pick Call Type & Count</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Choose which calls you want in the report.</p>', unsafe_allow_html=True)

    duration_config = get_duration_config(client_name)
    
    with st.expander(f"📊 Duration Configuration for {client_name}", expanded=False):
        has_xl = duration_config.get('extra_large_enabled', False)
        large_line = (
            f"• Large calls: {fmt_hms(duration_config['large_min'])} – {fmt_hms(duration_config['extra_large_min'])}<br>"
            f"• Extra Large calls: &gt; {fmt_hms(duration_config['extra_large_min'])}<br>"
            if has_xl else
            f"• Large calls: &gt; {fmt_hms(duration_config['large_min'])}<br>"
        )
        st.markdown(f"""
        <div class="config-box">
            <strong>Current Duration Settings:</strong><br>
            • Short calls: &lt; {fmt_hms(duration_config['short_max'])}<br>
            • Medium calls: {fmt_hms(duration_config['medium_min'])} – {fmt_hms(duration_config['medium_max'])}<br>
            {large_line}
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("Change the cutoffs below for this client (this browser session only), then click Apply.")
        oc1, oc2, oc3 = st.columns([1, 1, 1])
        with oc1:
            short_cutoff_min = st.number_input(
                "Short ends at (minutes)",
                min_value=0.5, max_value=15.0,
                value=round(duration_config['short_max'] / 60, 2),
                step=0.5,
                key=f"short_cutoff_{client_name}",
            )
        with oc2:
            large_cutoff_min = st.number_input(
                "Large starts after (minutes)",
                min_value=short_cutoff_min, max_value=30.0,
                value=max(round(duration_config['large_min'] / 60, 2), short_cutoff_min),
                step=0.5,
                key=f"large_cutoff_{client_name}",
            )
        with oc3:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                apply_clicked = st.button("✅ Apply", key="apply_overrides", use_container_width=True)
            with bcol2:
                if st.button("↩️ Reset", key="reset_overrides", use_container_width=True):
                    st.session_state["duration_overrides"].pop(client_name, None)
                    st.success("↩️ Reset to default.")
                    st.rerun()

        st.markdown("---")
        enable_xl = st.checkbox(
            "➕ Also split out an 'Extra Large' bucket (e.g. calls over 8 min)",
            value=duration_config.get('extra_large_enabled', False),
            key=f"enable_xl_{client_name}",
        )
        xl_cutoff_min = None
        if enable_xl:
            default_xl = duration_config.get('extra_large_min', duration_config['large_min'] + 180)
            xl_cutoff_min = st.number_input(
                "Extra Large starts after (minutes)",
                min_value=large_cutoff_min, max_value=60.0,
                value=max(round(default_xl / 60, 2), large_cutoff_min + 0.5),
                step=0.5,
                key=f"xl_cutoff_{client_name}",
            )

        if apply_clicked:
            override = {
                "short_max": int(round(short_cutoff_min * 60)),
                "medium_min": int(round(short_cutoff_min * 60)),
                "medium_max": int(round(large_cutoff_min * 60)),
                "large_min": int(round(large_cutoff_min * 60)),
                "extra_large_enabled": enable_xl,
            }
            if enable_xl and xl_cutoff_min:
                override["extra_large_min"] = int(round(xl_cutoff_min * 60))
            st.session_state["duration_overrides"][client_name] = override
            st.success("✅ Applied for this session!")
            st.rerun()

    st.markdown("#### 🔍 Search")
    scol1, scol2 = st.columns([1.6, 1.2])
    with scol1:
        agent_options = sorted(cdr_df[col_agent].dropna().astype(str).unique().tolist()) if col_agent else []
        selected_agents = st.multiselect(
            "Agent Name (leave empty for all agents)",
            options=agent_options,
            default=[],
        )
    with scol2:
        phone_search = st.text_input(
            "Phone Number contains",
            value="",
            placeholder="e.g. 98765",
        )

    if selected_agents and col_agent:
        cdr_df = cdr_df[cdr_df[col_agent].astype(str).isin(selected_agents)]
    if phone_search.strip() and col_phone:
        cdr_df = cdr_df[cdr_df[col_phone].astype(str).str.contains(phone_search.strip(), case=False, na=False)]

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f'<div class="metric-pill"><div class="value">{len(cdr_df)}</div><div class="label">Total valid calls</div></div>', unsafe_allow_html=True)
    with p2:
        avg_dur = cdr_df["_duration_sec"].mean() if len(cdr_df) else 0
        st.markdown(f'<div class="metric-pill"><div class="value">{fmt_hms(avg_dur)}</div><div class="label">Average call duration</div></div>', unsafe_allow_html=True)
    with p3:
        st.markdown(f'<div class="metric-pill"><div class="value">{client_name}</div><div class="label">Client</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown(f'<div class="metric-pill"><div class="value">{st.session_state.get("vdcl_removed", 0)}</div><div class="label">Abandoned (VDCL) removed</div></div>', unsafe_allow_html=True)

    bcol, ccol = st.columns([2, 1.2])
    with bcol:
        has_xl = duration_config.get('extra_large_enabled', False)
        short_label = f"Short (< {fmt_hms(duration_config['short_max'])})"
        medium_label = f"Medium ({fmt_hms(duration_config['medium_min'])} – {fmt_hms(duration_config['medium_max'])})"
        if has_xl:
            large_label = f"Large ({fmt_hms(duration_config['large_min'])} – {fmt_hms(duration_config['extra_large_min'])})"
            extra_large_label = f"Extra Large (> {fmt_hms(duration_config['extra_large_min'])})"
        else:
            large_label = f"Large (> {fmt_hms(duration_config['large_min'])})"
            extra_large_label = None

        bucket_options = ["All calls", short_label, medium_label, large_label]
        if has_xl:
            bucket_options.append(extra_large_label)
        bucket_options.append("Custom Filter")

        bucket = st.radio(
            "Call type",
            bucket_options,
            horizontal=True,
        )
        
        custom_filter_expr = ""
        if bucket == "Custom Filter":
            st.caption("📝 Pick calls by duration — no code needed.")
            filter_kind = st.radio(
                "Show calls where duration is...",
                ["Less than", "Greater than", "Between", "Exactly 0 (zero-duration)"],
                horizontal=True,
                key="custom_filter_kind",
            )
            if filter_kind == "Less than":
                cf_max = st.number_input("...this many minutes", min_value=0.0, value=2.0, step=0.5, key="cf_lt")
                custom_filter_expr = f"duration < {cf_max * 60}"
            elif filter_kind == "Greater than":
                cf_min = st.number_input("...this many minutes", min_value=0.0, value=8.0, step=0.5, key="cf_gt")
                custom_filter_expr = f"duration > {cf_min * 60}"
            elif filter_kind == "Between":
                cfb1, cfb2 = st.columns(2)
                with cfb1:
                    cf_from = st.number_input("From (minutes)", min_value=0.0, value=5.0, step=0.5, key="cf_from")
                with cfb2:
                    cf_to = st.number_input("To (minutes)", min_value=cf_from, value=10.0, step=0.5, key="cf_to")
                custom_filter_expr = f"duration >= {cf_from * 60} and duration <= {cf_to * 60}"
            else:
                custom_filter_expr = "duration == 0"
            st.caption(f"Filter applied: calls between the values you chose above.")
    
    with ccol:
        count_mode = st.radio("How many calls?", ["All matching", "Manual number"], horizontal=True)

    sort_order = st.radio(
        "Sort by Duration",
        ["Descending (longest first)", "Ascending (shortest first)"],
        horizontal=True,
        index=0,
    )
    ascending_sort = sort_order.startswith("Ascending")

    if bucket == short_label:
        matched = cdr_df[cdr_df["_duration_sec"] < duration_config['short_max']]
    elif bucket == medium_label:
        matched = cdr_df[(cdr_df["_duration_sec"] >= duration_config['medium_min']) & (cdr_df["_duration_sec"] <= duration_config['medium_max'])]
    elif bucket == large_label:
        if has_xl:
            matched = cdr_df[(cdr_df["_duration_sec"] > duration_config['large_min']) & (cdr_df["_duration_sec"] <= duration_config['extra_large_min'])]
        else:
            matched = cdr_df[cdr_df["_duration_sec"] > duration_config['large_min']]
    elif has_xl and bucket == extra_large_label:
        matched = cdr_df[cdr_df["_duration_sec"] > duration_config['extra_large_min']]
    elif bucket == "Custom Filter":
        if custom_filter_expr:
            expr = custom_filter_expr.replace('duration', 'cdr_df["_duration_sec"]')
            try:
                matched = cdr_df[eval(expr)]
            except Exception as e:
                st.error(f"⚠️ Error in filter expression: {e}")
                matched = cdr_df
        else:
            matched = cdr_df
    else:
        matched = cdr_df

    available = len(matched)
    if count_mode == "Manual number":
        manual_n = st.number_input("Number of calls", min_value=1, value=min(50, available) if available else 1, step=1)
        selected_df = matched.head(int(manual_n))
        if available < manual_n:
            st.warning(f"Only {available} call(s) match – showing all.")
        else:
            st.info(f"Showing {int(manual_n)} of {available} matching calls.")
    else:
        selected_df = matched
        st.info(f"Showing all {available} matching calls for **{bucket}**.")

    display_cols = [
        "campaign_id", "agent", "full_name", "leadid", "phone_number",
        "call_date", "start_time", "end_time", "call_duration1", "Recording", "_duration_sec"
    ]
    available_display_cols = [c for c in display_cols if c in selected_df.columns]
    table_df = selected_df[available_display_cols].copy()
    table_df.sort_values("_duration_sec", ascending=ascending_sort, inplace=True)
    table_df.insert(1, "S.No", range(1, len(table_df) + 1))
    final_display_cols = ["_duration_sec", "S.No"] + [c for c in available_display_cols if c != "_duration_sec"]
    table_df = table_df[final_display_cols]

    st.markdown(f"### Filtered Data – Sorted by Duration ({'ascending' if ascending_sort else 'descending'})")
    st.dataframe(table_df, use_container_width=True, height=350)

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # STEP 3 — RUN VAD + SENTIMENT ANALYSIS (Filtered Calls)
    # ============================================================
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">3</span>Run Talk-Time & Sentiment Analysis (Filtered Calls)</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Downloads recordings, measures speech/silence, transcribes with Groq Whisper, and performs sentiment analysis via Groq LLM.</p>', unsafe_allow_html=True)

    with st.expander("⚙️ Fine-tune detection accuracy (optional)"):
        st.caption(
            "If Talk Time is coming out too low / Silence too high, move the slider "
            "towards 'Detect more speech'. Default is fine for most calls."
        )
        sensitivity = st.slider(
            "Detection sensitivity",
            min_value=1, max_value=9, value=5,
            help="Lower = detect more speech. Higher = stricter, only counts confident speech.",
        )
        vad_threshold = round(0.15 + (sensitivity - 1) * (0.45 - 0.15) / 8, 3)
        dead_air_secs = st.number_input(
            "Count a pause as 'Dead Air' only if longer than (sec)",
            min_value=1, value=5, step=1,
        )

    run_vad_clicked = st.button("▶️  Run Analysis & Build Report (Filtered)", type="primary")

    if run_vad_clicked:
        if not col_recording:
            st.error("No recording-URL column found in CDR data — cannot fetch recordings.")
        elif len(selected_df) == 0:
            st.warning("No calls selected — nothing to process.")
        else:
            sentiment_pipeline = load_sentiment_pipeline()
            model, utils = load_vad_model()
            get_speech_timestamps = utils[0]

            VAD_CFG = {
                "threshold": vad_threshold,
                "min_speech_duration_ms": 100,
                "min_silence_duration_ms": 200,
                "speech_pad_ms": 300,
                "window_size_samples": 512,
                "dead_air_threshold_sec": dead_air_secs,
            }

            def run_vad_audio(audio_tensor):
                return get_speech_timestamps(
                    audio_tensor, model, sampling_rate=16000,
                    threshold=VAD_CFG["threshold"],
                    min_speech_duration_ms=VAD_CFG["min_speech_duration_ms"],
                    min_silence_duration_ms=VAD_CFG["min_silence_duration_ms"],
                    speech_pad_ms=VAD_CFG["speech_pad_ms"],
                    window_size_samples=VAD_CFG["window_size_samples"],
                )

            def compute_metrics_local(intervals, total_duration):
                if not intervals:
                    return {
                        "talk_time": 0.0,
                        "silence_time": round(total_duration, 2),
                        "dead_air": round(total_duration, 2) if total_duration > VAD_CFG["dead_air_threshold_sec"] else 0.0,
                        "longest_silence": round(total_duration, 2),
                    }
                speech_time, longest_silence, dead_air, prev_end = 0.0, 0.0, 0.0, 0.0
                for s, e in intervals:
                    speech_time += (e - s)
                    silence = max(0.0, s - prev_end)
                    longest_silence = max(longest_silence, silence)
                    if silence > VAD_CFG["dead_air_threshold_sec"]:
                        dead_air += silence
                    prev_end = e
                ending_silence = max(0.0, total_duration - prev_end)
                longest_silence = max(longest_silence, ending_silence)
                if ending_silence > VAD_CFG["dead_air_threshold_sec"]:
                    dead_air += ending_silence
                silence_time = max(0.0, total_duration - speech_time)
                return {
                    "talk_time": round(speech_time, 2),
                    "silence_time": round(silence_time, 2),
                    "dead_air": round(dead_air, 2),
                    "longest_silence": round(longest_silence, 2),
                }

            results = []
            progress = st.progress(0)
            status = st.empty()
            total_rows = len(selected_df)

            with tempfile.TemporaryDirectory() as tmpdir:
                for i, (_, row) in enumerate(selected_df.iterrows()):
                    status.text(f"Processing call {i+1}/{total_rows}...")
                    rec_url = row.get(col_recording)
                    metrics = {
                        "talk_time": None, "silence_time": None,
                        "dead_air": None, "longest_silence": None, "duration": None,
                    }
                    sentiment = "N/A"
                    transcript = None
                    debug_status = "OK"
                    actual_mp3 = None
                    mp3_path = os.path.join(tmpdir, f"{i}.mp3")
                    wav_path = os.path.join(tmpdir, f"{i}.wav")

                    try:
                        if not rec_url:
                            debug_status = "No recording URL in this row"
                        else:
                            actual_mp3 = resolve_audio_url(rec_url)
                            if not actual_mp3:
                                debug_status = "Could not resolve a direct audio URL from the recording link"
                            else:
                                r = requests.get(actual_mp3, timeout=60, stream=True)
                                if r.status_code != 200:
                                    debug_status = f"Download failed: HTTP {r.status_code}"
                                else:
                                    with open(mp3_path, "wb") as f:
                                        for chunk in r.iter_content(8192):
                                            if chunk:
                                                f.write(chunk)
                                    if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
                                        debug_status = "Downloaded file is empty"
                                    else:
                                        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
                                        ff = subprocess.run(
                                            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", wav_path],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        )
                                        if ff.returncode != 0 or not os.path.exists(wav_path):
                                            err_tail = ff.stderr.decode(errors="ignore")[-300:]
                                            debug_status = f"FFmpeg conversion failed: {err_tail.strip()}"
                                        else:
                                            data, sr = sf.read(wav_path)
                                            total_duration = len(data) / sr
                                            is_stereo = data.ndim > 1 and data.shape[1] > 1
                                            if is_stereo:
                                                all_intervals = []
                                                for ch in range(data.shape[1]):
                                                    tensor = load_channel_16k(data, sr, channel_idx=ch)
                                                    ts = run_vad_audio(tensor)
                                                    all_intervals.extend([(s["start"] / 16000, s["end"] / 16000) for s in ts])
                                                merged = merge_intervals(all_intervals)
                                            else:
                                                tensor = load_channel_16k(data, sr)
                                                ts = run_vad_audio(tensor)
                                                merged = merge_intervals([(s["start"] / 16000, s["end"] / 16000) for s in ts])
                                            metrics = compute_metrics_local(merged, total_duration)
                                            metrics["duration"] = round(total_duration, 2)
                                            
                                            if GROQ_API_KEY:
                                                try:
                                                    transcript = groq_transcribe(mp3_path, GROQ_API_KEY)
                                                    sentiment = analyze_sentiment(transcript, sentiment_pipeline)
                                                except Exception as e:
                                                    debug_status = f"Groq/Sentiment error: {str(e)[:100]}"
                                                    transcript = None
                                                    sentiment = "Error"
                                                else:
                                                    debug_status = "OK"
                                            else:
                                                sentiment = "No API Key"
                                                debug_status = "No API Key"
                    except requests.exceptions.RequestException as e:
                        debug_status = f"Network/download error: {str(e)[:100]}"
                    except Exception as e:
                        debug_status = f"Processing error: {str(e)[:100]}"
                    finally:
                        if os.path.exists(mp3_path):
                            try: os.remove(mp3_path)
                            except Exception: pass
                        if os.path.exists(wav_path):
                            try: os.remove(wav_path)
                            except Exception: pass

                    if debug_status != "OK" and debug_status != "No API Key" and "Groq" not in debug_status:
                        st.warning(f"Row {i+1} ({row.get(col_agent) if col_agent else ''}): {debug_status}")

                    crm_duration = row.get("_duration_sec")

                    date_val = row.get(col_date) if col_date else None
                    time_val = row.get(col_time) if col_time else None
                    if date_val and pd.notna(date_val):
                        try:
                            date_val = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                        except:
                            pass

                    results.append({
                        "Date": date_val,
                        "Time": time_val,
                        "Agent Name": row.get(col_agent) if col_agent else None,
                        "Call From": row.get(col_phone) if col_phone else None,
                        "Actual MP3": actual_mp3,
                        "Audio Duration(sec)": metrics.get("duration"),
                        "Audio Call Duration": crm_duration,
                        "AI Tools Talk time": metrics.get("talk_time"),
                        "Silence Time": metrics.get("silence_time"),
                        "Dead Air(included in Silence time)": metrics.get("dead_air"),
                        "Longest Silence": metrics.get("longest_silence"),
                        "Sentiment": sentiment,
                        "_debug_status": debug_status,
                    })
                    progress.progress((i + 1) / total_rows)

            status.text("Done ✅")
            final_df = pd.DataFrame(results)

            final_df.sort_values("Audio Call Duration", ascending=False, inplace=True)

            REORDERED_COLUMNS = [
                "Audio Call Duration",
                "Date",
                "Time",
                "Agent Name",
                "Call From",
                "AI Tools Talk time",
                "Silence Time",
                "Dead Air(included in Silence time)",
                "Longest Silence",
                "Sentiment",
                "Audio Duration(sec)",
                "Actual MP3",
            ]
            final_df = final_df[REORDERED_COLUMNS + ["_debug_status"]]

            agent_analytics_df = generate_agent_analytics(selected_df, '_duration_sec', duration_config)
            
            st.session_state["final_df"] = final_df
            st.session_state["agent_analytics_df"] = agent_analytics_df

            failed_count = (final_df["_debug_status"] != "OK").sum()
            if failed_count > 0:
                st.error(f"⚠️ {failed_count} of {len(final_df)} call(s) failed to process.")
            else:
                st.success("✅ All calls processed successfully.")

            st.dataframe(final_df.drop(columns=["_debug_status"]), use_container_width=True, height=380)
            
            if agent_analytics_df is not None and len(agent_analytics_df) > 0:
                st.markdown("### 📊 Agent-Wise Analytics")
                st.info("Comprehensive breakdown of call performance by agent.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏆 Top Agents by Large Calls**")
                    top_large = agent_analytics_df.nlargest(3, 'Large_Calls')[['Agent', 'Large_Calls', 'Large_%']]
                    for _, row in top_large.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">
                                {row['Large_Calls']} large calls ({row['Large_%']:.1f}% of total)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📉 Agents with Most Short Calls**")
                    top_short = agent_analytics_df.nlargest(3, 'Short_Calls')[['Agent', 'Short_Calls', 'Short_%']]
                    for _, row in top_short.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">
                                {row['Short_Calls']} short calls ({row['Short_%']:.1f}% of total)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("**📈 Performance Summary**")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Agents", len(agent_analytics_df))
                with m2:
                    avg_calls = agent_analytics_df['Total_Calls'].mean()
                    st.metric("Avg Calls per Agent", f"{avg_calls:.1f}")
                with m3:
                    best_agent = agent_analytics_df.iloc[0]['Agent'] if len(agent_analytics_df) > 0 else "N/A"
                    st.metric("Most Active Agent", best_agent)
                with m4:
                    best_large = agent_analytics_df.nlargest(1, 'Large_Calls')['Agent'].iloc[0] if len(agent_analytics_df) > 0 else "N/A"
                    st.metric("Most Large Calls", best_large)
                
                detail_cols = [
                    'Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%',
                    'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%',
                ]
                if 'Extra_Large_Calls' in agent_analytics_df.columns:
                    detail_cols += ['Extra_Large_Calls', 'Extra_Large_%']
                detail_cols += ['Avg_Duration_Formatted', 'Total_Duration_Formatted']
                st.dataframe(
                    agent_analytics_df[detail_cols],
                    use_container_width=True,
                    height=300
                )

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # STEP 4 — DOWNLOAD FINAL REPORT
    # ============================================================
    if st.session_state.get("final_df") is not None:
        EXPORT_COLUMNS = [
            "Audio Call Duration",
            "Date",
            "Time",
            "Agent Name",
            "Call From",
            "AI Tools Talk time",
            "Silence Time",
            "Dead Air(included in Silence time)",
            "Longest Silence",
            "Sentiment",
            "Audio Duration(sec)",
            "Actual MP3",
        ]
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-badge">4</span>Download Report</div>', unsafe_allow_html=True)
        st.markdown('<p class="step-subtitle">Excel file with two sheets: Detailed Call Report (now with Sentiment) and Agent-Wise Analytics.</p>', unsafe_allow_html=True)

        export_df = st.session_state["final_df"][EXPORT_COLUMNS]
        agent_export_df = st.session_state.get("agent_analytics_df")
        
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Call Report")
            if agent_export_df is not None and len(agent_export_df) > 0:
                has_xl_col = 'Extra_Large_Calls' in agent_export_df.columns
                export_cols = [
                    'Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%',
                    'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%',
                ]
                export_headers = [
                    'Rank', 'Agent', 'Total Calls', 'Short Calls', 'Short %',
                    'Medium Calls', 'Medium %', 'Large Calls', 'Large %',
                ]
                if has_xl_col:
                    export_cols += ['Extra_Large_Calls', 'Extra_Large_%']
                    export_headers += ['Extra Large Calls', 'Extra Large %']
                export_cols += ['Avg_Duration_Formatted', 'Total_Duration_Formatted']
                export_headers += ['Avg Duration', 'Total Duration']

                agent_sheet = agent_export_df[export_cols].copy()
                agent_sheet.columns = export_headers
                agent_sheet.to_excel(writer, index=False, sheet_name="Agent Analytics")
        buf.seek(0)
        st.download_button(
            "⬇️  Download Excel Report",
            data=buf,
            file_name=f"CallAI_Talk_Time_Report_{client_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # 🆕 STEP 5 — EMAIL MANAGEMENT + COMPLETE ANALYSIS
    # ============================================================
    st.markdown('<div class="step-card" style="border: 2px solid #4F46E5; background: #F5F4FF;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-title"><span class="step-badge" style="background:#7C3AED;">5</span>'
        '📧 Email Management & Auto-Schedule</div>',
        unsafe_allow_html=True,
    )
    
    # ---- Auto-Schedule Status ----
    st.markdown("### ⏰ Auto-Schedule Status")
    st.markdown(f"""
    <div class="schedule-box">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:16px;font-weight:600;">📅 Daily Schedule</span><br>
                <span style="color:#6B7280;">Runs every day at</span>
                <span class="time">11:00 PM</span>
                <span style="color:#6B7280;">for</span>
                <span style="font-weight:600;">{', '.join(AUTO_SCHEDULE_CLIENTS)}</span>
            </div>
            <div>
                <span style="background:#D1FAE5;color:#065F46;padding:6px 16px;border-radius:20px;font-weight:600;">
                    ✅ Active
                </span>
            </div>
        </div>
        <div style="margin-top:8px;font-size:13px;color:#6B7280;">
            Last run: {st.session_state.get("last_email_time", "Not run yet")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Email Management Section ----
    st.markdown("### 📧 Email Recipients Management")
    st.markdown("Add or remove mentor emails who will receive reports and alerts.")
    
    # Display current emails
    col_email1, col_email2 = st.columns([2, 1])
    with col_email1:
        st.markdown("**Current Recipients:**")
        if st.session_state.mentor_emails:
            email_html = ""
            for email in st.session_state.mentor_emails:
                email_html += f'<span class="email-chip">{email}</span>'
            st.markdown(f'<div class="email-list-box">{email_html}</div>', unsafe_allow_html=True)
        else:
            st.warning("No emails configured. Add at least one email to send reports.")
    
    with col_email2:
        # Add new email
        new_email = st.text_input("Add Email", placeholder="mentor@example.com")
        if st.button("➕ Add Email", use_container_width=True):
            if new_email and "@" in new_email and "." in new_email:
                if new_email not in st.session_state.mentor_emails:
                    st.session_state.mentor_emails.append(new_email.strip())
                    st.success(f"✅ Added {new_email}")
                    st.rerun()
                else:
                    st.warning("Email already exists!")
            else:
                st.error("Please enter a valid email address.")
        
        # Remove email dropdown
        if st.session_state.mentor_emails:
            remove_email = st.selectbox("Remove Email", [""] + st.session_state.mentor_emails)
            if st.button("🗑️ Remove Selected", use_container_width=True):
                if remove_email and remove_email in st.session_state.mentor_emails:
                    st.session_state.mentor_emails.remove(remove_email)
                    st.success(f"✅ Removed {remove_email}")
                    st.rerun()
        
        # Reset to default
        if st.button("↩️ Reset to Default", use_container_width=True):
            st.session_state.mentor_emails = DEFAULT_MENTOR_EMAILS.copy()
            st.success("✅ Reset to default emails")
            st.rerun()
    
    st.markdown("---")
    
    # ---- Test Email Section ----
    test_email_sending()
    
    st.markdown("---")
    
    # ---- Complete Analysis Section ----
    st.markdown("### 📊 Complete Call Analysis")
    st.markdown("Run VAD on ALL calls to analyze agent performance and detect defaulters.")
    
    total_calls = len(cdr_df)
    st.info(f"📞 **{total_calls}** calls available for complete analysis")

    st.markdown("**⚙️ Analysis Configuration**")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        silence_threshold = st.number_input(
            "Silence threshold (%)",
            min_value=20, max_value=50, value=30, step=5,
            help="Agents with overall OR short-call silence above this are flagged.",
            key="complete_silence_threshold"
        )
    with cfg_col2:
        min_calls_per_agent = st.number_input(
            "Minimum calls per agent",
            min_value=1, max_value=10, value=3, step=1,
            help="Agents with fewer calls are skipped.",
            key="complete_min_calls"
        )
    with cfg_col3:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**🎙️ Voice Detection Settings**")
    vad_col1, vad_col2 = st.columns(2)
    with vad_col1:
        complete_sensitivity = st.slider("VAD Sensitivity", 1, 9, 5, key="complete_sensitivity")
        complete_vad_threshold = round(0.15 + (complete_sensitivity - 1) * (0.45 - 0.15) / 8, 3)
    with vad_col2:
        complete_dead_air = st.number_input(
            "Dead-air threshold (sec)", min_value=1, value=5, step=1, key="complete_dead_air"
        )

    # ---- Buttons with status ----
    col_btn1, col_btn2, col_btn3 = st.columns([1.2, 1.2, 1])
    
    with col_btn1:
        run_complete_clicked = st.button(
            "▶️  Run Analysis", 
            type="primary", 
            key="run_complete_analysis_btn",
            use_container_width=True,
        )
    
    with col_btn2:
        # Send Performance Report Button with Status
        perf_btn = st.button(
            "📊 Send Performance Report",
            key="send_perf_report_btn",
            use_container_width=True,
            disabled=not st.session_state.get("agent_silence_by_category_df") is not None,
        )
        
    with col_btn3:
        def_btn = st.button(
            "🚨 Send Defaulter Alert",
            key="send_def_alert_btn",
            use_container_width=True,
            disabled=not st.session_state.get("defaulter_agents_df") is not None,
        )

    # ---- Run Analysis ----
    if run_complete_clicked:
        if not col_recording:
            st.error("No recording-URL column found — cannot run analysis.")
        elif len(cdr_df) == 0:
            st.warning("No calls to analyze.")
        else:
            dur_config = get_duration_config(client_name)

            with st.spinner("Running complete call analysis on ALL calls..."):
                silence_df, perf_df, defaulters_df = run_complete_call_analysis(
                    cdr_df, col_recording, col_agent, col_date, col_time, col_phone,
                    complete_vad_threshold, complete_dead_air, dur_config,
                    silence_threshold, min_calls_per_agent
                )

            st.session_state["complete_analysis_df"] = silence_df
            st.session_state["agent_silence_by_category_df"] = perf_df
            st.session_state["defaulter_agents_df"] = defaulters_df

            # ---------- Display Call-Level Data ----------
            st.markdown("### 📞 Call-Level Silence Analysis")
            st.info(f"Analyzed {len(silence_df)} calls across all categories")
            
            display_cols = ['Date', 'Time', 'Agent Name', 'Duration (sec)', 'Silence %', 'Longest Silence (sec)']
            display_cols = [c for c in display_cols if c in silence_df.columns]
            st.dataframe(silence_df[display_cols].head(20), use_container_width=True, height=300)

            # ---------- Performance Dashboard ----------
            if perf_df is not None and len(perf_df) > 0:
                st.markdown("### 📊 Agent Performance Dashboard")
                perf_display = perf_df[[
                    'Rank', 'Agent', 'Overall_Avg_Silence_%',
                    'Avg_Silence_%_Short', 'Avg_Silence_%_Medium', 'Avg_Silence_%_Large',
                    'Total_Calls'
                ]].copy()
                perf_display.columns = [
                    'Rank', 'Agent', 'Overall Silence %',
                    'Short %', 'Medium %', 'Large %', 'Total Calls'
                ]
                st.dataframe(perf_display, use_container_width=True, height=300)

            # ---------- Defaulter Detection ----------
            if defaulters_df is not None and len(defaulters_df) > 0:
                st.markdown(
                    f'<div class="status-banner-danger">🚨 {len(defaulters_df)} defaulter agent(s) detected!</div>',
                    unsafe_allow_html=True
                )
                
                def_display = defaulters_df[[
                    'Rank', 'Agent', 'Total_Calls',
                    'Short_Calls', 'Short_Silence_%',
                    'Medium_Calls', 'Medium_Silence_%',
                    'Large_Calls', 'Large_Silence_%',
                    'Overall_Silence_%'
                ]].copy()
                def_display.columns = [
                    'Rank', 'Agent', 'Total Calls',
                    'Short Calls', 'Short Silence %',
                    'Medium Calls', 'Medium Silence %',
                    'Large Calls', 'Large Silence %',
                    'Overall Silence %'
                ]
                st.dataframe(def_display, use_container_width=True, height=300)

    # ---- Send Performance Report ----
    if perf_btn:
        perf_df = st.session_state.get("agent_silence_by_category_df")
        silence_df = st.session_state.get("complete_analysis_df")
        
        if perf_df is None or len(perf_df) == 0:
            st.error("❌ Please run the analysis first!")
        elif not st.session_state.mentor_emails:
            st.error("❌ No mentor emails configured! Please add emails.")
        else:
            with st.spinner("📧 Sending performance report..."):
                success, message = send_performance_report_email(
                    perf_df, 
                    silence_df, 
                    client_name, 
                    st.session_state.mentor_emails
                )
                
                if success:
                    st.session_state["last_email_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"""
                    <div class="email-status-box email-success">
                        ✅ {message}
                        <br><small>Sent at: {st.session_state['last_email_time']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="email-status-box email-error">
                        ❌ {message}
                    </div>
                    """, unsafe_allow_html=True)

    # ---- Send Defaulter Alert ----
    if def_btn:
        defaulters_df = st.session_state.get("defaulter_agents_df")
        silence_df = st.session_state.get("complete_analysis_df")
        
        if defaulters_df is None or len(defaulters_df) == 0:
            st.error("❌ No defaulters found! Run analysis first or check your threshold.")
        elif not st.session_state.mentor_emails:
            st.error("❌ No mentor emails configured! Please add emails.")
        else:
            with st.spinner("📧 Sending defaulter alert..."):
                success, message = send_defaulter_alert_email(
                    defaulters_df, 
                    silence_df, 
                    client_name, 
                    st.session_state.mentor_emails
                )
                
                if success:
                    st.session_state["last_email_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"""
                    <div class="email-status-box email-success">
                        ✅ {message}
                        <br><small>Sent at: {st.session_state['last_email_time']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="email-status-box email-error">
                        ❌ {message}
                    </div>
                    """, unsafe_allow_html=True)

    # ---------- Downloads ----------
    st.markdown("### ⬇️ Download Complete Analysis Reports")
    
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        if st.session_state.get("complete_analysis_df") is not None:
            silence_df = st.session_state["complete_analysis_df"]
            buf_s = io.BytesIO()
            with pd.ExcelWriter(buf_s, engine="openpyxl") as writer:
                silence_df.to_excel(writer, index=False, sheet_name="Call Data")
            buf_s.seek(0)
            st.download_button(
                "⬇️ Download Call Data",
                data=buf_s,
                file_name=f"Complete_Call_Data_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_data",
            )
    with dcol2:
        if st.session_state.get("agent_silence_by_category_df") is not None:
            perf_df = st.session_state["agent_silence_by_category_df"]
            buf_p = io.BytesIO()
            with pd.ExcelWriter(buf_p, engine="openpyxl") as writer:
                perf_df.to_excel(writer, index=False, sheet_name="Performance Dashboard")
            buf_p.seek(0)
            st.download_button(
                "⬇️ Download Performance Report",
                data=buf_p,
                file_name=f"Performance_Dashboard_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_perf",
            )
    with dcol3:
        if st.session_state.get("defaulter_agents_df") is not None and len(st.session_state["defaulter_agents_df"]) > 0:
            defaulters_df = st.session_state["defaulter_agents_df"]
            buf_d = io.BytesIO()
            with pd.ExcelWriter(buf_d, engine="openpyxl") as writer:
                defaulters_df.to_excel(writer, index=False, sheet_name="Defaulter Agents")
            buf_d.seek(0)
            st.download_button(
                "⬇️ Download Defaulter Report",
                data=buf_d,
                file_name=f"Defaulter_Report_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_def",
            )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Pick a client and date range above, then click **Fetch Calls** to get started.")
