# -*- coding: utf-8 -*-
"""
CallAI Analytics - Streamlit App (SaaS Edition)
================================================
Fetches CDR call data from the CRM, filters out abandoned (VDCL) calls,
lets you pick a call-type bucket, runs Silero VAD for talk-time/silence
metrics, transcribes calls with Groq Whisper and scores sentiment with a
Groq LLM, and produces downloadable Excel reports plus emailed
performance/defaulter alerts for mentors.

Requires: streamlit >= 1.32 (uses st.dialog for the Email Portal).
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
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import librosa
from bs4 import BeautifulSoup
import streamlit as st
import torch
import torchaudio
import groq

# Try to import silero_vad, fallback to torch.hub if not available
try:
    from silero_vad import load_silero_vad, VADIterator
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    print("silero_vad package not found, using torch.hub fallback")

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
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
    .status-banner-danger {
        background: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
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
    .email-list-box {
        background: #F8F7FF;
        border: 1px solid #E4E1FF;
        border-radius: 10px;
        padding: 12px 16px;
        min-height: 50px;
        margin-top: 8px;
    }
    .buffer-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 20px;
        background: #F5F4FF;
        border: 1px solid #E4E1FF;
        border-radius: 12px;
        margin: 10px 0;
    }
    .buffer-spinner {
        width: 20px;
        height: 20px;
        border: 3px solid #E4E1FF;
        border-top-color: #4F46E5;
        border-radius: 50%;
        animation: buffer-spin 0.8s linear infinite;
        flex-shrink: 0;
    }
    @keyframes buffer-spin { to { transform: rotate(360deg); } }
    .buffer-label {
        font-weight: 600;
        color: #4F46E5;
        font-size: 14px;
    }
    .success-box {
        background: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 12px 16px;
        color: #065F46;
        font-weight: 600;
        margin: 8px 0;
    }
    .warning-box {
        background: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 12px 16px;
        color: #92400E;
        font-weight: 600;
        margin: 8px 0;
    }
    .info-box {
        background: #DBEAFE;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 12px 16px;
        color: #1E40AF;
        font-weight: 600;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="callai-hero">
    <h1>📞 CallAI · Talk-Time + Sentiment</h1>
    <p>Pick a client, fetch calls, filter, get Talk-Time / Silence / Dead-Air, and analyse sentiment with Groq Whisper + Groq LLM.</p>
</div>
""", unsafe_allow_html=True)


def render_buffer(container, message="Working..."):
    """Show a small animated 'buffering' indicator in place of raw progress text."""
    container.markdown(
        f'<div class="buffer-box"><div class="buffer-spinner"></div>'
        f'<div class="buffer-label">{message}</div></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# CRM + Groq + SMTP configuration
# ----------------------------------------------------------------------
CRM_BASE = "https://crmapi.dialdesk.in"
LOGIN_URL = f"{CRM_BASE}/auth/login"
CDR_URL = f"{CRM_BASE}/report/cdr_report"

try:
    CRM_EMAIL = st.secrets["CRM_EMAIL"]
    CRM_PASSWORD = st.secrets["CRM_PASSWORD"]
except Exception:
    CRM_EMAIL = "ispark@dialdesk.in"
    CRM_PASSWORD = "1234"

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""


def _get_secret(key, default):
    """Read a value from st.secrets, falling back to `default` if the key is missing."""
    try:
        return st.secrets[key]
    except Exception:
        return default


# ----------------------------------------------------------------------
# SMTP Configuration - UPDATE THESE VALUES
# ----------------------------------------------------------------------
# Default SMTP Config (will be overridden by session state)
DEFAULT_SMTP_CONFIG = {
    "host": "mail.dialdesk.net",  # Try this first - updated from mail.domain.com
    "port": 587,
    "username": "tickets@dialdesk.net",
    "password": "DiaL@#433#212",
    "from_email": "tickets@dialdesk.net",
    "use_tls": True,
}

# Alternative SMTP configurations to try automatically
ALTERNATIVE_SMTP_CONFIGS = [
    # dialdesk.net servers
    {"host": "mail.dialdesk.net", "port": 587, "use_tls": True},
    {"host": "smtp.dialdesk.net", "port": 587, "use_tls": True},
    {"host": "mail.dialdesk.net", "port": 465, "use_tls": False},
    {"host": "smtp.dialdesk.net", "port": 465, "use_tls": False},
    {"host": "mail.dialdesk.net", "port": 25, "use_tls": False},
    
    # If your domain is actually domain.com
    {"host": "mail.domain.com", "port": 587, "use_tls": True},
    {"host": "smtp.domain.com", "port": 587, "use_tls": True},
    
    # Free SMTP services (you may need to sign up)
    # SendGrid - free tier (100 emails/day)
    {"host": "smtp.sendgrid.net", "port": 587, "use_tls": True, "username": "apikey", "password": ""},
    # Mailgun - free tier
    {"host": "smtp.mailgun.org", "port": 587, "use_tls": True, "username": "", "password": ""},
    # Gmail - if you have Gmail
    {"host": "smtp.gmail.com", "port": 587, "use_tls": True, "username": "", "password": ""},
]

DEFAULT_MENTOR_EMAILS = [
    "urvi.wadhwa@teammas.in",
]

AUTO_SCHEDULE_CLIENTS = ["Weebo", "Hari Om Pvt Ltd"]

CLIENTS = {
    "Weebo": "687",
    "Hari Om Pvt Ltd": "689",
    "F1 INFO SOLUTION": "609",
    "Saatvik": "663",
    "Fortum Charge": "395",
    "Alphanso": "629",
}

DEFAULT_DURATION_CONFIG = {
    "short_max": 120,
    "medium_min": 120,
    "medium_max": 300,
    "large_min": 300,
}

CLIENT_DURATION_CONFIGS = {
    "Weebo": {
        "short_max": 90,
        "medium_min": 90,
        "medium_max": 240,
        "large_min": 240,
    },
    "Hari Om Pvt Ltd": {
        "short_max": 150,
        "medium_min": 150,
        "medium_max": 360,
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


# ----------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------
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
    "last_email_time": None,
    "scheduler_running": False,
    "email_large_threshold_min": None,
    "smtp_config": DEFAULT_SMTP_CONFIG.copy(),
    "scheduler_time": "23:00",
    "scheduler_enabled": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_smtp_config():
    """Get current SMTP config from session state."""
    return st.session_state.get("smtp_config", DEFAULT_SMTP_CONFIG.copy())


def update_smtp_config(new_config):
    """Update SMTP config in session state."""
    st.session_state["smtp_config"] = new_config


# ----------------------------------------------------------------------
# CRM functions
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Recording download / resolution
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Column mapping + duration parsing
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Agent analytics (filtered-selection view, Step 3)
# ----------------------------------------------------------------------
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

    agent_stats['Short_Calls'] = agent_stats['Category_Counts'].apply(lambda x: extract_category_counts(x, 'Short'))
    agent_stats['Medium_Calls'] = agent_stats['Category_Counts'].apply(lambda x: extract_category_counts(x, 'Medium'))
    agent_stats['Large_Calls'] = agent_stats['Category_Counts'].apply(lambda x: extract_category_counts(x, 'Large'))
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


# ----------------------------------------------------------------------
# VAD and Audio Processing Functions
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_silero_vad_model():
    """Load Silero VAD model using torch.hub with caching."""
    box = st.empty()
    render_buffer(box, "Loading voice-detection model (first run only)...")

    # Set torch hub directory
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
    except Exception:
        hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub")
        os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)

    try:
        # Try loading from torch.hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        get_speech_timestamps = utils[0]
        box.empty()
        return model, get_speech_timestamps
    except Exception as e:
        print(f"Error loading VAD from torch.hub: {e}")
        box.empty()
        return None, None


def compute_speech_energy_based(audio, sr, threshold=0.01, min_speech_duration=0.1):
    """Simple energy-based speech detection as fallback."""
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    
    # Compute energy in windows
    window_size = int(sr * 0.025)  # 25ms windows
    hop_size = int(sr * 0.010)     # 10ms hop
    
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        energy.append(np.sqrt(np.mean(window**2)))
    
    energy = np.array(energy)
    
    # Find speech regions
    is_speech = energy > threshold
    min_frames = int(min_speech_duration * sr / hop_size)
    
    # Merge speech regions
    speech_intervals = []
    start = None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i * hop_size / sr
        elif not speech and start is not None:
            end = i * hop_size / sr
            if end - start >= min_speech_duration:
                speech_intervals.append((start, end))
            start = None
    
    if start is not None:
        end = len(audio) / sr
        if end - start >= min_speech_duration:
            speech_intervals.append((start, end))
    
    return speech_intervals


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95
    return audio


def process_audio_file(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load and process audio file, convert to mono if needed."""
    try:
        data, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        # Normalize
        data = normalize_audio(data)
        
        return data, sr
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None


def compute_vad_metrics(audio: np.ndarray, sr: int, threshold: float = 0.3, 
                        dead_air_secs: float = 5.0) -> Dict:
    """Compute VAD metrics using Silero VAD or fallback."""
    total_duration = len(audio) / sr
    
    try:
        # Try Silero VAD first
        model, get_speech_timestamps = load_silero_vad_model()
        
        if model is not None and get_speech_timestamps is not None:
            # Use Silero VAD
            audio_tensor = torch.from_numpy(audio).float()
            
            vad_kwargs = {
                'sampling_rate': sr,
                'threshold': threshold,
                'min_speech_duration_ms': 250,
                'min_silence_duration_ms': 200,
                'speech_pad_ms': 400,
                'window_size_samples': 512,
            }
            
            speech_timestamps = get_speech_timestamps(audio_tensor, model, **vad_kwargs)
            
            # Convert to seconds
            speech_intervals = []
            for ts in speech_timestamps:
                start = ts['start'] / sr
                end = ts['end'] / sr
                speech_intervals.append((start, end))
            
            # Calculate metrics
            return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)
        
    except Exception as e:
        print(f"Silero VAD failed, using fallback: {e}")
    
    # Fallback to energy-based detection
    speech_intervals = compute_speech_energy_based(audio, sr, threshold=0.02)
    return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)


def calculate_metrics_from_intervals(speech_intervals: List[Tuple[float, float]], 
                                    total_duration: float, 
                                    dead_air_secs: float) -> Dict:
    """Calculate talk time, silence, and dead air from speech intervals."""
    if not speech_intervals:
        return {
            "talk_time": 0.0,
            "silence_time": round(total_duration, 2),
            "dead_air": round(total_duration, 2) if total_duration > dead_air_secs else 0.0,
            "longest_silence": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "speech_segments": []
        }
    
    # Sort and merge overlapping intervals
    speech_intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in speech_intervals:
        if not merged or start > merged[-1][1] + 0.1:  # Small gap tolerance
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    
    # Calculate metrics
    speech_time = 0.0
    longest_silence = 0.0
    dead_air = 0.0
    prev_end = 0.0
    
    for start, end in merged:
        speech_time += (end - start)
        
        # Calculate silence before this segment
        silence = max(0.0, start - prev_end)
        longest_silence = max(longest_silence, silence)
        if silence > dead_air_secs:
            dead_air += silence
        prev_end = end
    
    # Final silence
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
        "duration": round(total_duration, 2),
        "speech_segments": [(round(s, 2), round(e, 2)) for s, e in merged]
    }


def transcribe_audio_chunk(client: groq.Groq, audio: np.ndarray, sr: int) -> str:
    """Transcribe a single audio chunk using Groq Whisper."""
    try:
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Create WAV in memory
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        data = buf.getvalue()
        buf = io.BytesIO(data)
        
        # Prompt for code-mixed Hindi-English
        prompt = '''
This conversation is between an agent of a company called Weebo and one of their customers. Weebo is an internet service provider. The call will be about internet services, internet problems, billing, etc.

The call will be code mixed Hindi and english. Bahut sare words Hindi mein honge.

Some frequent words : namaste, hello, namaskar, sir, ma'am, complain, internet, raha, rahi, deta, deti
'''
        
        result = client.audio.transcriptions.create(
            file=('audio.wav', buf),
            model='whisper-large-v3-turbo',
            prompt=prompt,
            response_format='text',
            language='hi',
        )
        
        return str(result)
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""


def process_recording_metrics_only(rec_url, vad_threshold, dead_air_secs, tmpdir, tag):
    """Process recording for metrics only (no transcription)."""
    metrics = {"talk_time": 0.0, "silence_time": 0.0, "dead_air": 0.0, 
               "longest_silence": 0.0, "duration": 0.0}
    debug_status = "OK"
    actual_mp3 = None
    mp3_path = os.path.join(tmpdir, f"{tag}.mp3")
    wav_path = os.path.join(tmpdir, f"{tag}.wav")

    try:
        if not rec_url:
            return metrics, "No recording URL in this row", None

        actual_mp3 = resolve_audio_url(rec_url)
        if not actual_mp3:
            return metrics, "Could not resolve audio URL", None

        # Download audio
        r = requests.get(actual_mp3, timeout=120, stream=True)
        if r.status_code != 200:
            return metrics, f"Download failed: HTTP {r.status_code}", actual_mp3
        
        with open(mp3_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            return metrics, "Downloaded file is empty", actual_mp3

        # Convert to WAV
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", 
             "-ac", "1", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        
        if ff.returncode != 0 or not os.path.exists(wav_path):
            err_tail = ff.stderr.decode(errors="ignore")[-300:]
            return metrics, f"FFmpeg conversion failed: {err_tail.strip()}", actual_mp3

        # Load audio
        audio_data, sr = process_audio_file(wav_path)
        if audio_data is None:
            return metrics, "Failed to load audio", actual_mp3

        # Compute VAD metrics
        metrics = compute_vad_metrics(audio_data, sr, vad_threshold, dead_air_secs)
        return metrics, "OK", actual_mp3

    except requests.exceptions.RequestException as e:
        return metrics, f"Network error: {str(e)[:100]}", actual_mp3
    except Exception as e:
        return metrics, f"Processing error: {str(e)[:100]}", actual_mp3
    finally:
        for path in [mp3_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


def process_recording_with_transcription(rec_url, vad_threshold, dead_air_secs, tmpdir, tag):
    """Process recording with full VAD + transcription."""
    metrics = {"talk_time": 0.0, "silence_time": 0.0, "dead_air": 0.0, 
               "longest_silence": 0.0, "duration": 0.0}
    debug_status = "OK"
    transcript = ""
    sentiment = "Neutral"
    actual_mp3 = None
    mp3_path = os.path.join(tmpdir, f"{tag}.mp3")
    wav_path = os.path.join(tmpdir, f"{tag}.wav")

    try:
        if not rec_url:
            return metrics, "No recording URL in this row", None, "", "Neutral"

        actual_mp3 = resolve_audio_url(rec_url)
        if not actual_mp3:
            return metrics, "Could not resolve audio URL", None, "", "Neutral"

        # Download audio
        r = requests.get(actual_mp3, timeout=120, stream=True)
        if r.status_code != 200:
            return metrics, f"Download failed: HTTP {r.status_code}", actual_mp3, "", "Neutral"
        
        with open(mp3_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            return metrics, "Downloaded file is empty", actual_mp3, "", "Neutral"

        # Convert to WAV
        ffmpeg_cmd = shutil.which("ffmpeg") or "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "16000", 
             "-ac", "1", wav_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        
        if ff.returncode != 0 or not os.path.exists(wav_path):
            err_tail = ff.stderr.decode(errors="ignore")[-300:]
            return metrics, f"FFmpeg conversion failed: {err_tail.strip()}", actual_mp3, "", "Neutral"

        # Load audio
        audio_data, sr = process_audio_file(wav_path)
        if audio_data is None:
            return metrics, "Failed to load audio", actual_mp3, "", "Neutral"

        # Compute VAD metrics
        vad_result = compute_vad_metrics(audio_data, sr, vad_threshold, dead_air_secs)
        metrics = vad_result
        
        # If we have API key and speech was detected, transcribe
        if GROQ_API_KEY and vad_result.get("talk_time", 0) > 1.0:
            try:
                # Get speech segments from VAD result
                speech_segments = vad_result.get("speech_segments", [])
                if speech_segments:
                    client = groq.Groq(api_key=GROQ_API_KEY)
                    transcript_parts = []
                    
                    for start, end in speech_segments[:5]:  # Limit to first 5 segments
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        segment = audio_data[start_sample:end_sample]
                        
                        if len(segment) > sr * 0.5:  # At least 0.5 seconds
                            text = transcribe_audio_chunk(client, segment, sr)
                            if text:
                                transcript_parts.append(text)
                    
                    transcript = "\n\n".join(transcript_parts)
                    
                    # Analyze sentiment if we have transcript
                    if transcript:
                        sentiment = analyze_sentiment(transcript)
                else:
                    # Try transcribing the whole audio if no segments detected
                    if len(audio_data) > sr * 1.0:  # At least 1 second
                        client = groq.Groq(api_key=GROQ_API_KEY)
                        transcript = transcribe_audio_chunk(client, audio_data, sr)
                        if transcript:
                            sentiment = analyze_sentiment(transcript)
                
                debug_status = "OK"
                
            except Exception as e:
                debug_status = f"Transcription error: {str(e)[:100]}"
        else:
            if not GROQ_API_KEY:
                debug_status = "No API Key"
            else:
                debug_status = "No speech detected"

        return metrics, debug_status, actual_mp3, transcript, sentiment

    except requests.exceptions.RequestException as e:
        return metrics, f"Network error: {str(e)[:100]}", actual_mp3, "", "Neutral"
    except Exception as e:
        return metrics, f"Processing error: {str(e)[:100]}", actual_mp3, "", "Neutral"
    finally:
        for path in [mp3_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of transcribed text using Groq LLM."""
    if not text or not text.strip():
        return "Neutral"
    if not GROQ_API_KEY:
        return "No API Key"
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


# ----------------------------------------------------------------------
# Complete call analysis (all calls) 
# ----------------------------------------------------------------------
def detect_defaulters_simple(silence_df, duration_config, silence_threshold=30, min_calls=3):
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

        is_defaulter = (overall_avg_silence > silence_threshold) or (short_avg_silence > silence_threshold)

        if is_defaulter:
            agent_data.append({
                'Agent': agent,
                'Total_Calls': total_calls,
                'Short_Calls': len(short_calls),
                'Short_Silence_%': round(short_avg_silence, 1),
                'Medium_Calls': len(medium_calls),
                'Medium_Silence_%': round(medium_avg_silence, 1),
                'Large_Calls': len(large_calls),
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
    """3-bucket (Short/Medium/Large) breakdown used for the on-screen dashboard."""
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
        index='Agent', columns='Category', values=['Avg_Silence_%', 'Call_Count'], fill_value=0
    ).reset_index()

    pivot_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_df.columns.values]
    pivot_df = pivot_df.rename(columns={'Agent_': 'Agent'})

    for cat in ['Short', 'Medium', 'Large']:
        if f'Avg_Silence_%_{cat}' not in pivot_df.columns:
            pivot_df[f'Avg_Silence_%_{cat}'] = 0
        if f'Call_Count_{cat}' not in pivot_df.columns:
            pivot_df[f'Call_Count_{cat}'] = 0

    agent_overall = silence_df_copy.groupby('Agent Name').agg({
        'Silence %': 'mean', 'Silence Time (sec)': 'sum', 'Duration (sec)': 'sum'
    }).reset_index()
    agent_overall.columns = ['Agent', 'Overall_Avg_Silence_%', 'Total_Silence_Time', 'Total_Duration']

    pivot_df = pivot_df.merge(agent_overall, on='Agent', how='left')
    pivot_df['Total_Calls'] = pivot_df['Call_Count_Short'] + pivot_df['Call_Count_Medium'] + pivot_df['Call_Count_Large']
    pivot_df = pivot_df.sort_values('Overall_Avg_Silence_%', ascending=False).reset_index(drop=True)
    pivot_df['Rank'] = range(1, len(pivot_df) + 1)

    return pivot_df


def categorize_duration_4(duration, short_max, medium_max, large_threshold_sec):
    """Short / Medium / Long / Large bucket used for the emailed performance report."""
    if duration is None or pd.isna(duration):
        return "Unknown"
    if duration < short_max:
        return "Short"
    if duration <= medium_max:
        return "Medium"
    if duration <= large_threshold_sec:
        return "Long"
    return "Large"


def build_agent_performance_table(silence_df, short_max, medium_max, large_threshold_sec):
    """Per-agent Short/Medium/Long/Large call counts with average silence % for each bucket."""
    if silence_df is None or len(silence_df) == 0:
        return None

    df = silence_df.copy()
    df["Duration (sec)"] = pd.to_numeric(df["Duration (sec)"], errors="coerce")
    df["Category"] = df["Duration (sec)"].apply(
        lambda d: categorize_duration_4(d, short_max, medium_max, large_threshold_sec)
    )
    df = df[df["Category"] != "Unknown"]
    if len(df) == 0:
        return None

    rows = []
    for agent, g in df.groupby("Agent Name"):
        row = {"Agent": agent, "Total_Calls": len(g), "Overall_Silence": round(g["Silence %"].mean(), 1)}
        for cat in ["Short", "Medium", "Long", "Large"]:
            sub = g[g["Category"] == cat]
            row[f"{cat}_Calls"] = len(sub)
            row[f"{cat}_Silence"] = round(sub["Silence %"].mean(), 1) if len(sub) else 0.0
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Overall_Silence", ascending=False).reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out


def default_large_threshold_min(duration_config):
    return round(duration_config["medium_max"] / 60 + 3, 1)


def run_complete_call_analysis(all_calls_df, col_recording, col_agent, col_date, col_time, col_phone,
                                vad_threshold, dead_air_secs, duration_config,
                                silence_threshold=30, min_calls=3):
    """Runs VAD on ALL calls, returns call-level data, the on-screen performance table, and defaulters."""
    progress = st.progress(0)
    buffer_box = st.empty()
    render_buffer(buffer_box, "Analyzing calls...")
    total = len(all_calls_df)
    silence_rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (_, row) in enumerate(all_calls_df.iterrows()):
            metrics, debug_status, actual_mp3 = process_recording_metrics_only(
                row.get(col_recording), vad_threshold, dead_air_secs, tmpdir, f"complete_{i}",
            )
            dur = metrics.get("duration")
            sil = metrics.get("silence_time")
            silence_pct = round((sil / dur * 100), 1) if dur and dur > 0 and sil is not None else None

            date_val = row.get(col_date) if col_date else None
            time_val = row.get(col_time) if col_time else None

            if date_val and pd.notna(date_val):
                try:
                    date_val = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                except Exception:
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

    buffer_box.empty()
    progress.empty()
    silence_df = pd.DataFrame(silence_rows)

    perf_df = analyze_agent_silence_by_category(silence_df, duration_config)
    defaulters_df = detect_defaulters_simple(silence_df, duration_config, silence_threshold, min_calls)

    return silence_df, perf_df, defaulters_df


# ----------------------------------------------------------------------
# Email functions with configurable SMTP - IMPROVED
# ----------------------------------------------------------------------
def test_smtp_connection(smtp_config=None):
    """Test SMTP connection with given config."""
    if smtp_config is None:
        smtp_config = get_smtp_config()
    
    try:
        if not smtp_config.get("password"):
            return False, "SMTP password not configured."

        host = smtp_config.get("host", "mail.dialdesk.net")
        port = int(smtp_config.get("port", 587))
        username = smtp_config.get("username", "")
        password = smtp_config.get("password", "")
        use_tls = smtp_config.get("use_tls", True)

        print(f"🔍 Testing SMTP connection to {host}:{port} with TLS={use_tls}")
        print(f"   Username: {username}")
        
        # Try to connect with timeout
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            server.login(username, password)

        return True, f"✅ SMTP connection successful to {host}:{port}"
    except smtplib.SMTPAuthenticationError as e:
        return False, f"❌ Authentication failed: {str(e)}\nCheck username/password."
    except smtplib.SMTPServerDisconnected as e:
        return False, f"❌ Server disconnected: {str(e)}\nServer may not support SMTP."
    except ConnectionRefusedError:
        return False, f"❌ Connection refused. Server {host}:{port} is not responding.\nCheck if SMTP service is running on this server."
    except TimeoutError:
        return False, f"❌ Connection timed out to {host}:{port}.\nCheck network/firewall settings."
    except Exception as e:
        error_msg = str(e)
        if "timed out" in error_msg.lower() or "10060" in error_msg:
            return False, f"❌ Connection timed out to {host}:{port}.\n\nPossible solutions:\n1. The server '{host}' might be incorrect\n2. Try using the server's IP address instead\n3. Check if port {port} is open for SMTP\n4. Check firewall settings\n5. Try port 465 (SSL) or 25 (no encryption)"
        return False, f"❌ SMTP Error: {error_msg}"


def try_alternative_smtp_configs():
    """Try alternative SMTP configurations and return the working one."""
    print("🔄 Trying alternative SMTP configurations...")
    
    for i, alt_config in enumerate(ALTERNATIVE_SMTP_CONFIGS, 1):
        # Skip configurations without username/password for services that need them
        if not alt_config.get("username") or not alt_config.get("password"):
            # For services that need special auth, skip if no credentials
            if alt_config.get("host") in ["smtp.sendgrid.net", "smtp.mailgun.org"]:
                continue
        
        print(f"   Testing #{i}: {alt_config['host']}:{alt_config['port']}")
        ok, msg = test_smtp_connection(alt_config)
        if ok:
            print(f"   ✅ Found working SMTP: {alt_config['host']}:{alt_config['port']}")
            return alt_config, msg
        else:
            print(f"   ❌ Failed: {msg[:100]}")
    
    return None, "No working SMTP configuration found. Please check your settings."


def _send_mime(msg, mentor_emails, smtp_config=None):
    """Send MIME message using configured SMTP."""
    if smtp_config is None:
        smtp_config = get_smtp_config()
    
    host = smtp_config.get("host", "mail.dialdesk.net")
    port = int(smtp_config.get("port", 587))
    username = smtp_config.get("username", "")
    password = smtp_config.get("password", "")
    use_tls = smtp_config.get("use_tls", True)
    
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        server.login(username, password)
        server.sendmail(msg["From"], mentor_emails, msg.as_string())


def send_performance_report_email(perf_table, silence_df, client_name, mentor_emails, large_threshold_min, smtp_config=None):
    """Send performance report email."""
    if smtp_config is None:
        smtp_config = get_smtp_config()
    
    if perf_table is None or len(perf_table) == 0:
        return False, "No performance data available"

    mentor_emails = [e.strip() for e in mentor_emails if e and '@' in e]
    if not mentor_emails:
        return False, "No valid mentor emails found!"

    # Test SMTP connection first
    smtp_ok, smtp_msg = test_smtp_connection(smtp_config)
    if not smtp_ok:
        # Try alternative configurations
        alt_config, alt_msg = try_alternative_smtp_configs()
        if alt_config:
            smtp_config = alt_config
            update_smtp_config(smtp_config)
            st.session_state["smtp_config"] = smtp_config
            print(f"Using alternative SMTP: {alt_config}")
        else:
            return False, f"SMTP Error: {smtp_msg}\n\nTried all configurations. Please check:\n1. Server address\n2. Port number\n3. Username/Password\n4. Network connectivity"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config.get("from_email") or smtp_config["username"]
        msg["To"] = ", ".join(mentor_emails)
        msg["Subject"] = f"📊 Agent Performance Report — {client_name} ({date.today().strftime('%d/%m/%Y')})"

        agent_rows = ""
        for _, row in perf_table.iterrows():
            overall = row.get('Overall_Silence', 0)
            color = '#DC2626' if overall > 40 else ('#D97706' if overall > 30 else '#16A34A')
            agent_rows += f"""
            <tr>
                <td style="padding:10px;font-weight:bold;">{row.get('Agent', 'Unknown')}</td>
                <td style="padding:10px;text-align:center;">{row.get('Total_Calls', 0)}</td>
                <td style="padding:10px;text-align:center;">{row.get('Short_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Short_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Medium_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Medium_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Long_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Long_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Large_Calls', 0)}<br><small style="color:#6B7280;">({row.get('Large_Silence', 0)}%)</small></td>
                <td style="padding:10px;text-align:center;font-weight:bold;color:{color};">{overall}%</td>
            </tr>
            """

        avg_overall = perf_table['Overall_Silence'].mean() if 'Overall_Silence' in perf_table else 0
        best_agent = perf_table.iloc[0].get('Agent', 'N/A') if len(perf_table) > 0 else "N/A"
        worst_agent = perf_table.iloc[-1].get('Agent', 'N/A') if len(perf_table) > 0 else "N/A"
        total_agents = len(perf_table)

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
            <p style="color:#6B7280;font-size:13px;">
                "Large" = calls longer than {large_threshold_min:.1f} min · counts shown with average silence % in brackets.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Total Calls</th>
                        <th>Short<br><small>(Silence %)</small></th>
                        <th>Medium<br><small>(Silence %)</small></th>
                        <th>Long<br><small>(Silence %)</small></th>
                        <th>Large<br><small>(Silence %)</small></th>
                        <th>Overall %</th>
                    </tr>
                </thead>
                <tbody>
                    {agent_rows}
                </tbody>
            </table>

            <div class="footer">
                <p>— CallAI Analytics | Automated Performance Report</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                perf_table.to_excel(writer, index=False, sheet_name="Performance Report")
                if silence_df is not None and len(silence_df) > 0:
                    silence_df.to_excel(writer, index=False, sheet_name="All Calls Data")
            buf.seek(0)
            excel_attachment = MIMEApplication(buf.read(), _subtype="xlsx")
            excel_attachment.add_header(
                'Content-Disposition', 'attachment',
                filename=f'Performance_Report_{client_name}_{date.today().strftime("%Y%m%d")}.xlsx'
            )
            msg.attach(excel_attachment)
        except Exception as e:
            print(f"Excel attachment error: {e}")

        try:
            _send_mime(msg, mentor_emails, smtp_config)
            return True, f"✅ Performance report sent to {len(mentor_emails)} mentors! (Using {smtp_config['host']}:{smtp_config['port']})"
        except Exception as e:
            return False, f"❌ Failed to send email: {str(e)}"

    except Exception as e:
        return False, f"❌ Error preparing email: {str(e)}"


def send_defaulter_alert_email(defaulters_df, silence_df, client_name, mentor_emails, smtp_config=None):
    """Send defaulter alert email."""
    if smtp_config is None:
        smtp_config = get_smtp_config()
    
    if defaulters_df is None or len(defaulters_df) == 0:
        return False, "No defaulters to report"

    mentor_emails = [e.strip() for e in mentor_emails if e and '@' in e]
    if not mentor_emails:
        return False, "No valid mentor emails found!"

    smtp_ok, smtp_msg = test_smtp_connection(smtp_config)
    if not smtp_ok:
        # Try alternative configurations
        alt_config, alt_msg = try_alternative_smtp_configs()
        if alt_config:
            smtp_config = alt_config
            update_smtp_config(smtp_config)
            st.session_state["smtp_config"] = smtp_config
            print(f"Using alternative SMTP: {alt_config}")
        else:
            return False, f"SMTP Error: {smtp_msg}\n\nTried all configurations. Please check:\n1. Server address\n2. Port number\n3. Username/Password\n4. Network connectivity"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config.get("from_email") or smtp_config["username"]
        msg["To"] = ", ".join(mentor_emails)
        msg["Subject"] = f"🚨 Defaulter Alert — {len(defaulters_df)} agent(s) flagged ({client_name})"

        agent_rows = ""
        for _, row in defaulters_df.iterrows():
            overall = row.get('Overall_Silence_%', 0)
            if overall > 40:
                color, badge = '#DC2626', '🔴 HIGH'
            elif overall > 30:
                color, badge = '#D97706', '🟠 MEDIUM'
            else:
                color, badge = '#2563EB', '🔵 LOW'

            agent_rows += f"""
            <tr>
                <td style="padding:10px;font-weight:bold;color:{color};">{row.get('Agent', 'Unknown')}</td>
                <td style="padding:10px;text-align:center;">{row.get('Total_Calls', 0)}</td>
                <td style="padding:10px;text-align:center;">{row.get('Short_Calls', 0)}<br><small>{row.get('Short_Silence_%', 0)}%</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Medium_Calls', 0)}<br><small>{row.get('Medium_Silence_%', 0)}%</small></td>
                <td style="padding:10px;text-align:center;">{row.get('Large_Calls', 0)}<br><small>{row.get('Large_Silence_%', 0)}%</small></td>
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
                <p>Agents with <strong>Overall Silence &gt; 30%</strong> OR <strong>Short Call Silence &gt; 30%</strong> are flagged.</p>
                <p><span style="color:#DC2626;">🔴 High</span> = &gt;40% |
                   <span style="color:#D97706;">🟠 Medium</span> = 30-40% |
                   <span style="color:#2563EB;">🔵 Low</span> = &lt;30%</p>
            </div>
            <h3>📊 Defaulter Agents</h3>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th><th>Total Calls</th>
                        <th>Short<br><small style="font-weight:normal;">(Silence)</small></th>
                        <th>Medium<br><small style="font-weight:normal;">(Silence)</small></th>
                        <th>Large<br><small style="font-weight:normal;">(Silence)</small></th>
                        <th>Overall %</th><th>Severity</th>
                    </tr>
                </thead>
                <tbody>{agent_rows}</tbody>
            </table>
            <div class="footer">
                <p>— CallAI Analytics | Automated Defaulter Detection</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        try:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                defaulters_df.to_excel(writer, index=False, sheet_name="Defaulters")
                if silence_df is not None and len(silence_df) > 0:
                    silence_df.to_excel(writer, index=False, sheet_name="All Calls Data")
            buf.seek(0)
            excel_attachment = MIMEApplication(buf.read(), _subtype="xlsx")
            excel_attachment.add_header(
                'Content-Disposition', 'attachment',
                filename=f'Defaulter_Alert_{client_name}_{date.today().strftime("%Y%m%d")}.xlsx'
            )
            msg.attach(excel_attachment)
        except Exception as e:
            print(f"Excel attachment error: {e}")

        try:
            _send_mime(msg, mentor_emails, smtp_config)
            return True, f"✅ Defaulter alert sent to {len(mentor_emails)} mentors! (Using {smtp_config['host']}:{smtp_config['port']})"
        except Exception as e:
            return False, f"❌ Failed to send email: {str(e)}"

    except Exception as e:
        return False, f"❌ Error preparing email: {str(e)}"


# ----------------------------------------------------------------------
# Auto-scheduler with configurable time
# ----------------------------------------------------------------------
def run_scheduled_job():
    """Run the scheduled job for all auto-schedule clients."""
    try:
        # Get current SMTP config from session state
        smtp_config = st.session_state.get("smtp_config", DEFAULT_SMTP_CONFIG.copy())
        
        do_login()
        yesterday = date.today() - timedelta(days=1)

        for client_name in AUTO_SCHEDULE_CLIENTS:
            if client_name not in CLIENTS:
                continue

            payload = {
                "from_date": yesterday.strftime("%Y-%m-%d"),
                "to_date": yesterday.strftime("%Y-%m-%d"),
                "company_id": str(CLIENTS[client_name]),
            }

            resp = fetch_cdr(payload)
            if resp.status_code != 200:
                continue

            data = resp.json()
            records = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(records, dict):
                for v in records.values():
                    if isinstance(v, list):
                        records = v
                        break
            cdr_df = pd.DataFrame(records)
            cdr_df, _ = filter_out_vdcl_calls(cdr_df)
            _, duration_seconds = resolve_duration_column(cdr_df)
            cdr_df["_duration_sec"] = duration_seconds

            if len(cdr_df) == 0:
                continue

            col_recording = find_column(cdr_df, COLUMN_CANDIDATES["recording"])
            col_agent = find_column(cdr_df, COLUMN_CANDIDATES["agent_name"])
            col_date = find_column(cdr_df, COLUMN_CANDIDATES["date"])
            col_time = find_column(cdr_df, COLUMN_CANDIDATES["time"])
            col_phone = find_column(cdr_df, COLUMN_CANDIDATES["call_from"])

            if not col_recording:
                continue

            duration_config = get_duration_config(client_name)
            silence_df, _, defaulters_df = run_complete_call_analysis(
                cdr_df, col_recording, col_agent, col_date, col_time, col_phone,
                vad_threshold=0.3, dead_air_secs=5, duration_config=duration_config,
                silence_threshold=30, min_calls=3
            )

            mentor_emails = st.session_state.get("mentor_emails", DEFAULT_MENTOR_EMAILS)
            large_threshold_sec = default_large_threshold_min(duration_config) * 60
            perf_table = build_agent_performance_table(
                silence_df, duration_config["short_max"], duration_config["medium_max"], large_threshold_sec
            )

            if perf_table is not None and len(perf_table) > 0:
                send_performance_report_email(
                    perf_table, silence_df, client_name, mentor_emails,
                    default_large_threshold_min(duration_config), smtp_config
                )

            if defaulters_df is not None and len(defaulters_df) > 0:
                send_defaulter_alert_email(defaulters_df, silence_df, client_name, mentor_emails, smtp_config)

        st.session_state["last_email_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        print(f"[{datetime.now()}] Scheduled job error: {str(e)}")


def start_scheduler():
    """Start the scheduler with configured time."""
    if st.session_state.get("scheduler_running", False):
        return
    
    scheduler_time = st.session_state.get("scheduler_time", "23:00")
    scheduler_enabled = st.session_state.get("scheduler_enabled", True)
    
    if not scheduler_enabled:
        return

    def scheduler_loop():
        # Clear any existing schedule
        schedule.clear()
        schedule.every().day.at(scheduler_time).do(run_scheduled_job)
        while True:
            schedule.run_pending()
            time.sleep(60)

    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    st.session_state["scheduler_running"] = True


def restart_scheduler():
    """Restart the scheduler with new settings."""
    st.session_state["scheduler_running"] = False
    time.sleep(1)
    start_scheduler()


# ----------------------------------------------------------------------
# Email & Reports Portal (modal dialog) - IMPROVED
# ----------------------------------------------------------------------
@st.dialog("📧 Email & Reports Portal")
def open_email_portal(client_name):
    st.caption("Manage recipients, SMTP settings, and send reports.")

    # --- SMTP Configuration Section ---
    st.markdown("### 📧 SMTP Configuration")
    st.caption("Configure your email server settings. These will be used for all email reports.")
    
    smtp_config = st.session_state.get("smtp_config", DEFAULT_SMTP_CONFIG.copy())
    
    with st.expander("⚙️ SMTP Settings", expanded=True):
        st.markdown("""
        <div class="info-box">
        💡 <strong>Important:</strong> <code>mail.domain.com</code> is a placeholder. 
        You need to use your actual SMTP server. For <code>tickets@dialdesk.net</code>, try:
        <ul>
            <li><code>mail.dialdesk.net</code> (Port 587, TLS)</li>
            <li><code>smtp.dialdesk.net</code> (Port 587, TLS)</li>
            <li><code>mail.dialdesk.net</code> (Port 465, SSL)</li>
        </ul>
        If unsure, contact your email hosting provider.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            new_host = st.text_input("SMTP Server", value=smtp_config.get("host", "mail.dialdesk.net"), 
                                    help="e.g., smtp.gmail.com, mail.yourdomain.com", key="smtp_host")
            new_port = st.number_input("Port", value=int(smtp_config.get("port", 587)), step=1, 
                                      help="587 (TLS) or 465 (SSL) or 25", key="smtp_port")
            new_username = st.text_input("Username/Email", value=smtp_config.get("username", "tickets@dialdesk.net"), 
                                        key="smtp_username")
        with col2:
            new_password = st.text_input("Password", value=smtp_config.get("password", ""), type="password", key="smtp_password")
            new_from_email = st.text_input("From Email", value=smtp_config.get("from_email", "tickets@dialdesk.net"), 
                                          key="smtp_from")
            use_tls = st.checkbox("Use TLS/SSL", value=smtp_config.get("use_tls", True), 
                                help="Enable for ports 587 and 465", key="smtp_tls")
        
        # Common SMTP settings quick reference
        st.markdown("**🔧 Common SMTP Settings:**")
        st.markdown("""
        | Service | Server | Port | TLS/SSL |
        |---------|--------|------|---------|
        | Gmail | smtp.gmail.com | 587 | TLS |
        | Gmail (SSL) | smtp.gmail.com | 465 | SSL |
        | Office 365 | smtp.office365.com | 587 | TLS |
        | SendGrid | smtp.sendgrid.net | 587 | TLS |
        | Mailgun | smtp.mailgun.org | 587 | TLS |
        | Custom Domain | mail.yourdomain.com | 587 | TLS |
        | Custom Domain (SSL) | mail.yourdomain.com | 465 | SSL |
        """)
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("💾 Save Settings", key="save_smtp"):
                updated_config = {
                    "host": new_host,
                    "port": new_port,
                    "username": new_username,
                    "password": new_password,
                    "from_email": new_from_email or new_username,
                    "use_tls": use_tls,
                }
                update_smtp_config(updated_config)
                st.success("✅ SMTP settings saved!")
                st.rerun()
        
        with col_btn2:
            if st.button("🔬 Test Connection", key="test_smtp_btn"):
                ok, smtp_msg = test_smtp_connection()
                if ok:
                    st.markdown(f'<div class="success-box">✅ {smtp_msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warning-box">⚠️ {smtp_msg}</div>', unsafe_allow_html=True)
        
        with col_btn3:
            if st.button("🔄 Auto-Find SMTP", key="try_alt_smtp"):
                with st.spinner("Trying alternative SMTP configurations..."):
                    alt_config, alt_msg = try_alternative_smtp_configs()
                    if alt_config:
                        update_smtp_config(alt_config)
                        st.success(f"✅ Found working SMTP: {alt_config['host']}:{alt_config['port']}")
                        st.rerun()
                    else:
                        st.error(f"❌ {alt_msg}")

    st.markdown("---")
    
    # --- Auto-Scheduler Configuration ---
    st.markdown("### ⏰ Auto-Scheduler")
    st.caption("Configure when daily reports should be sent automatically.")
    
    col1, col2 = st.columns(2)
    with col1:
        current_time = st.session_state.get("scheduler_time", "23:00")
        new_scheduler_time = st.text_input("Report Time (HH:MM)", value=current_time, key="scheduler_time_input")
    with col2:
        scheduler_enabled = st.checkbox("Enable Auto-Scheduler", value=st.session_state.get("scheduler_enabled", True), key="scheduler_enabled_check")
    
    if st.button("🔄 Update Scheduler", key="update_scheduler"):
        # Validate time format
        try:
            datetime.strptime(new_scheduler_time, "%H:%M")
            st.session_state["scheduler_time"] = new_scheduler_time
            st.session_state["scheduler_enabled"] = scheduler_enabled
            restart_scheduler()
            st.success(f"✅ Scheduler updated! Reports will run daily at {new_scheduler_time}")
        except ValueError:
            st.error("❌ Invalid time format. Please use HH:MM (e.g., 23:00)")

    if st.session_state.get("scheduler_running", False):
        st.markdown(f'<div class="success-box">🟢 Scheduler is running. Next report at {st.session_state.get("scheduler_time", "23:00")}</div>', unsafe_allow_html=True)
    else:
        status_text = "enabled" if st.session_state.get("scheduler_enabled", True) else "disabled"
        st.markdown(f'<div class="warning-box">🟡 Scheduler is {status_text}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- Recipients Management ---
    st.markdown("### 👥 Recipients")
    st.caption("Manage email recipients for reports.")
    
    if st.session_state.mentor_emails:
        chips = "".join(f'<span class="email-chip">{e}</span>' for e in st.session_state.mentor_emails)
        st.markdown(f'<div class="email-list-box">{chips}</div>', unsafe_allow_html=True)
    else:
        st.warning("No recipients added yet.")

    c1, c2 = st.columns([2, 1])
    with c1:
        new_email = st.text_input("Add recipient", placeholder="mentor@example.com", key="portal_new_email")
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("➕ Add", use_container_width=True, key="portal_add_email"):
            if new_email and "@" in new_email and "." in new_email:
                if new_email.strip() not in st.session_state.mentor_emails:
                    st.session_state.mentor_emails.append(new_email.strip())
                    st.rerun()
            else:
                st.error("Enter a valid email address.")

    if st.session_state.mentor_emails:
        rcol1, rcol2, rcol3 = st.columns([2, 1, 1])
        with rcol1:
            remove_email = st.selectbox("Remove recipient", [""] + st.session_state.mentor_emails, key="portal_remove_select")
        with rcol2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Remove", use_container_width=True, key="portal_remove_btn"):
                if remove_email:
                    st.session_state.mentor_emails.remove(remove_email)
                    st.rerun()
        with rcol3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("↩️ Reset", use_container_width=True, key="portal_reset_btn"):
                st.session_state.mentor_emails = DEFAULT_MENTOR_EMAILS.copy()
                st.rerun()

    st.markdown("---")
    
    # --- Large Call Threshold ---
    duration_config = get_duration_config(client_name)
    current_min = round(duration_config["medium_max"] / 60, 1)
    stored = st.session_state.get("email_large_threshold_min")
    default_val = stored if stored and stored >= current_min else default_large_threshold_min(duration_config)

    st.markdown("**Large call threshold**")
    st.caption("Calls longer than this are counted as 'Large'; calls between Medium and this cutoff are 'Long'.")
    large_threshold_min = st.number_input(
        "Large calls start after (minutes)",
        min_value=current_min, max_value=60.0, value=default_val, step=0.5,
        key="portal_large_threshold",
    )
    st.session_state["email_large_threshold_min"] = large_threshold_min

    st.markdown("---")
    
    # --- Preview ---
    silence_df = st.session_state.get("complete_analysis_df")
    perf_table = None
    if silence_df is None or len(silence_df) == 0:
        st.info("Run **Complete Call Analysis** (Step 5) first to enable the report preview.")
    else:
        perf_table = build_agent_performance_table(
            silence_df, duration_config["short_max"], duration_config["medium_max"], large_threshold_min * 60
        )
        if perf_table is not None:
            st.markdown("**Preview**")
            preview = perf_table[[
                "Rank", "Agent", "Total_Calls",
                "Short_Calls", "Short_Silence",
                "Medium_Calls", "Medium_Silence",
                "Long_Calls", "Long_Silence",
                "Large_Calls", "Large_Silence",
                "Overall_Silence",
            ]]
            st.dataframe(preview, use_container_width=True, height=200)

    st.markdown("---")
    
    # --- Send Reports ---
    st.markdown("### 📤 Send Reports")
    scol1, scol2 = st.columns(2)
    defaulters_df = st.session_state.get("defaulter_agents_df")

    with scol1:
        send_perf = st.button(
            "📊 Send Performance Report", type="primary", use_container_width=True,
            disabled=perf_table is None, key="portal_send_perf",
        )
    with scol2:
        send_def = st.button(
            "🚨 Send Defaulter Alert", use_container_width=True,
            disabled=defaulters_df is None or len(defaulters_df) == 0, key="portal_send_def",
        )

    if send_perf:
        if not st.session_state.mentor_emails:
            st.error("Add at least one recipient first.")
        else:
            smtp_config = st.session_state.get("smtp_config", DEFAULT_SMTP_CONFIG.copy())
            with st.spinner("Sending performance report..."):
                ok, result_msg = send_performance_report_email(
                    perf_table, silence_df, client_name, st.session_state.mentor_emails, 
                    large_threshold_min, smtp_config
                )
            if ok:
                st.session_state["last_email_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(result_msg)
            else:
                st.error(result_msg)

    if send_def:
        if not st.session_state.mentor_emails:
            st.error("Add at least one recipient first.")
        else:
            smtp_config = st.session_state.get("smtp_config", DEFAULT_SMTP_CONFIG.copy())
            with st.spinner("Sending defaulter alert..."):
                ok, result_msg = send_defaulter_alert_email(
                    defaulters_df, silence_df, client_name, st.session_state.mentor_emails, smtp_config
                )
            if ok:
                st.session_state["last_email_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success(result_msg)
            else:
                st.error(result_msg)

    # Last email status
    if st.session_state.get("last_email_time"):
        st.caption(f"📨 Last email sent: {st.session_state['last_email_time']}")


# ----------------------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------------------
# Start scheduler on app load
if not st.session_state.get("scheduler_running", False):
    start_scheduler()

# ---- Step 1: Client + date range ----
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

# ---- Step 2 onwards: only once we have data ----
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

    # ---- Step 2: filter & display ----
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
                "Short ends at (minutes)", min_value=0.5, max_value=15.0,
                value=round(duration_config['short_max'] / 60, 2), step=0.5,
                key=f"short_cutoff_{client_name}",
            )
        with oc2:
            large_cutoff_min = st.number_input(
                "Large starts after (minutes)", min_value=short_cutoff_min, max_value=30.0,
                value=max(round(duration_config['large_min'] / 60, 2), short_cutoff_min), step=0.5,
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
            value=duration_config.get('extra_large_enabled', False), key=f"enable_xl_{client_name}",
        )
        xl_cutoff_min = None
        if enable_xl:
            default_xl = duration_config.get('extra_large_min', duration_config['large_min'] + 180)
            xl_cutoff_min = st.number_input(
                "Extra Large starts after (minutes)", min_value=large_cutoff_min, max_value=60.0,
                value=max(round(default_xl / 60, 2), large_cutoff_min + 0.5), step=0.5,
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
        selected_agents = st.multiselect("Agent Name (leave empty for all agents)", options=agent_options, default=[])
    with scol2:
        phone_search = st.text_input("Phone Number contains", value="", placeholder="e.g. 98765")

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
        bucket = st.radio("Call type", bucket_options, horizontal=True)

        custom_filter_expr = ""
        if bucket == "Custom Filter":
            st.caption("📝 Pick calls by duration — no code needed.")
            filter_kind = st.radio(
                "Show calls where duration is...",
                ["Less than", "Greater than", "Between", "Exactly 0 (zero-duration)"],
                horizontal=True, key="custom_filter_kind",
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
            st.caption("Filter applied: calls between the values you chose above.")

    with ccol:
        count_mode = st.radio("How many calls?", ["All matching", "Manual number"], horizontal=True)

    sort_order = st.radio(
        "Sort by Duration", ["Descending (longest first)", "Ascending (shortest first)"], horizontal=True, index=0,
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

    # ---- Step 3: VAD + sentiment on filtered calls ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">3</span>Run Talk-Time & Sentiment Analysis (Filtered Calls)</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Downloads recordings, measures speech/silence, transcribes with Groq Whisper (VAD-based), and performs sentiment analysis via Groq LLM.</p>', unsafe_allow_html=True)

    with st.expander("⚙️ Fine-tune detection accuracy (optional)"):
        st.caption(
            "If Talk Time is coming out too low / Silence too high, move the slider "
            "towards 'Detect more speech'. Default is fine for most calls."
        )
        sensitivity = st.slider("Detection sensitivity", min_value=1, max_value=9, value=5,
                                 help="Lower = detect more speech. Higher = stricter, only counts confident speech.")
        vad_threshold = round(0.1 + (sensitivity - 1) * (0.35 - 0.1) / 8, 3)
        dead_air_secs = st.number_input("Count a pause as 'Dead Air' only if longer than (sec)", min_value=1, value=5, step=1)

    run_vad_clicked = st.button("▶️  Run Analysis & Build Report (Filtered)", type="primary")

    if run_vad_clicked:
        if not col_recording:
            st.error("No recording-URL column found in CDR data — cannot fetch recordings.")
        elif len(selected_df) == 0:
            st.warning("No calls selected — nothing to process.")
        else:
            results = []
            progress = st.progress(0)
            buffer_box = st.empty()
            render_buffer(buffer_box, "Analyzing calls, transcribing & scoring sentiment...")
            total_rows = len(selected_df)

            with tempfile.TemporaryDirectory() as tmpdir:
                for i, (_, row) in enumerate(selected_df.iterrows()):
                    rec_url = row.get(col_recording)
                    
                    metrics, debug_status, actual_mp3, transcript, sentiment = process_recording_with_transcription(
                        rec_url, vad_threshold, dead_air_secs, tmpdir, f"filtered_{i}"
                    )
                    
                    if debug_status != "OK" and "API" not in debug_status:
                        st.warning(f"Row {i+1} ({row.get(col_agent) if col_agent else ''}): {debug_status}")

                    crm_duration = row.get("_duration_sec")
                    date_val = row.get(col_date) if col_date else None
                    time_val = row.get(col_time) if col_time else None
                    if date_val and pd.notna(date_val):
                        try:
                            date_val = pd.to_datetime(date_val).strftime("%d/%m/%Y")
                        except Exception:
                            pass

                    results.append({
                        "Date": date_val, "Time": time_val,
                        "Agent Name": row.get(col_agent) if col_agent else None,
                        "Call From": row.get(col_phone) if col_phone else None,
                        "Actual MP3": actual_mp3,
                        "Audio Duration(sec)": metrics.get("duration"),
                        "Audio Call Duration": crm_duration,
                        "AI Tools Talk time": metrics.get("talk_time"),
                        "Silence Time": metrics.get("silence_time"),
                        "Dead Air(included in Silence time)": metrics.get("dead_air"),
                        "Longest Silence": metrics.get("longest_silence"),
                        "Transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript,
                        "Sentiment": sentiment,
                        "_debug_status": debug_status,
                    })
                    progress.progress((i + 1) / total_rows)

            buffer_box.empty()
            progress.empty()

            final_df = pd.DataFrame(results)
            final_df.sort_values("Audio Call Duration", ascending=False, inplace=True)
            REORDERED_COLUMNS = [
                "Audio Call Duration", "Date", "Time", "Agent Name", "Call From",
                "AI Tools Talk time", "Silence Time", "Dead Air(included in Silence time)",
                "Longest Silence", "Transcript", "Sentiment", "Audio Duration(sec)", "Actual MP3",
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
                            <div class="agent-stats">{row['Large_Calls']} large calls ({row['Large_%']:.1f}% of total)</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("**📉 Agents with Most Short Calls**")
                    top_short = agent_analytics_df.nlargest(3, 'Short_Calls')[['Agent', 'Short_Calls', 'Short_%']]
                    for _, row in top_short.iterrows():
                        st.markdown(f"""
                        <div class="agent-card">
                            <div class="agent-name">{row['Agent']}</div>
                            <div class="agent-stats">{row['Short_Calls']} short calls ({row['Short_%']:.1f}% of total)</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("**📈 Performance Summary**")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Agents", len(agent_analytics_df))
                with m2:
                    st.metric("Avg Calls per Agent", f"{agent_analytics_df['Total_Calls'].mean():.1f}")
                with m3:
                    st.metric("Most Active Agent", agent_analytics_df.iloc[0]['Agent'] if len(agent_analytics_df) > 0 else "N/A")
                with m4:
                    best_large = agent_analytics_df.nlargest(1, 'Large_Calls')['Agent'].iloc[0] if len(agent_analytics_df) > 0 else "N/A"
                    st.metric("Most Large Calls", best_large)

                detail_cols = ['Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%', 'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%']
                if 'Extra_Large_Calls' in agent_analytics_df.columns:
                    detail_cols += ['Extra_Large_Calls', 'Extra_Large_%']
                detail_cols += ['Avg_Duration_Formatted', 'Total_Duration_Formatted']
                st.dataframe(agent_analytics_df[detail_cols], use_container_width=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 4: download filtered report ----
    if st.session_state.get("final_df") is not None:
        EXPORT_COLUMNS = [
            "Audio Call Duration", "Date", "Time", "Agent Name", "Call From",
            "AI Tools Talk time", "Silence Time", "Dead Air(included in Silence time)",
            "Longest Silence", "Transcript", "Sentiment", "Audio Duration(sec)", "Actual MP3",
        ]
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-title"><span class="step-badge">4</span>Download Report</div>', unsafe_allow_html=True)
        st.markdown('<p class="step-subtitle">Excel file with two sheets: Detailed Call Report (with Transcript & Sentiment) and Agent-Wise Analytics.</p>', unsafe_allow_html=True)

        export_df = st.session_state["final_df"][EXPORT_COLUMNS]
        agent_export_df = st.session_state.get("agent_analytics_df")

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Call Report")
            if agent_export_df is not None and len(agent_export_df) > 0:
                has_xl_col = 'Extra_Large_Calls' in agent_export_df.columns
                export_cols = ['Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_%', 'Medium_Calls', 'Medium_%', 'Large_Calls', 'Large_%']
                export_headers = ['Rank', 'Agent', 'Total Calls', 'Short Calls', 'Short %', 'Medium Calls', 'Medium %', 'Large Calls', 'Large %']
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
            "⬇️  Download Excel Report", data=buf,
            file_name=f"CallAI_Talk_Time_Report_{client_name.replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 5: Complete Call Analysis (all calls) ----
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-badge">5</span>Complete Call Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="step-subtitle">Run VAD on ALL calls to build the agent performance dashboard and detect defaulters.</p>', unsafe_allow_html=True)

    total_calls = len(cdr_df)
    st.info(f"📞 **{total_calls}** calls available for complete analysis")

    st.markdown("**⚙️ Analysis Configuration**")
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
    with cfg_col1:
        silence_threshold = st.number_input(
            "Silence threshold (%)", min_value=20, max_value=50, value=30, step=5,
            help="Agents with overall OR short-call silence above this are flagged.",
            key="complete_silence_threshold"
        )
    with cfg_col2:
        min_calls_per_agent = st.number_input(
            "Minimum calls per agent", min_value=1, max_value=10, value=3, step=1,
            help="Agents with fewer calls are skipped.", key="complete_min_calls"
        )
    with cfg_col3:
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**🎙️ Voice Detection Settings**")
    vad_col1, vad_col2 = st.columns(2)
    with vad_col1:
        complete_sensitivity = st.slider("VAD Sensitivity", 1, 9, 5, key="complete_sensitivity")
        complete_vad_threshold = round(0.1 + (complete_sensitivity - 1) * (0.35 - 0.1) / 8, 3)
    with vad_col2:
        complete_dead_air = st.number_input("Dead-air threshold (sec)", min_value=1, value=5, step=1, key="complete_dead_air")

    run_complete_clicked = st.button("▶️  Run Analysis", type="primary", key="run_complete_analysis_btn")

    if run_complete_clicked:
        if not col_recording:
            st.error("No recording-URL column found — cannot run analysis.")
        elif len(cdr_df) == 0:
            st.warning("No calls to analyze.")
        else:
            dur_config = get_duration_config(client_name)
            silence_df, perf_df, defaulters_df = run_complete_call_analysis(
                cdr_df, col_recording, col_agent, col_date, col_time, col_phone,
                complete_vad_threshold, complete_dead_air, dur_config,
                silence_threshold, min_calls_per_agent
            )
            st.session_state["complete_analysis_df"] = silence_df
            st.session_state["agent_silence_by_category_df"] = perf_df
            st.session_state["defaulter_agents_df"] = defaulters_df

            st.markdown("### 📞 Call-Level Silence Analysis")
            st.info(f"Analyzed {len(silence_df)} calls across all categories")
            display_cols = ['Date', 'Time', 'Agent Name', 'Duration (sec)', 'Silence %', 'Longest Silence (sec)']
            display_cols = [c for c in display_cols if c in silence_df.columns]
            st.dataframe(silence_df[display_cols].head(20), use_container_width=True, height=300)

            if perf_df is not None and len(perf_df) > 0:
                st.markdown("### 📊 Agent Performance Dashboard")
                perf_display = perf_df[[
                    'Rank', 'Agent', 'Overall_Avg_Silence_%',
                    'Avg_Silence_%_Short', 'Avg_Silence_%_Medium', 'Avg_Silence_%_Large', 'Total_Calls'
                ]].copy()
                perf_display.columns = ['Rank', 'Agent', 'Overall Silence %', 'Short %', 'Medium %', 'Large %', 'Total Calls']
                st.dataframe(perf_display, use_container_width=True, height=300)

            if defaulters_df is not None and len(defaulters_df) > 0:
                st.markdown(
                    f'<div class="status-banner-danger">🚨 {len(defaulters_df)} defaulter agent(s) detected!</div>',
                    unsafe_allow_html=True
                )
                def_display = defaulters_df[[
                    'Rank', 'Agent', 'Total_Calls', 'Short_Calls', 'Short_Silence_%',
                    'Medium_Calls', 'Medium_Silence_%', 'Large_Calls', 'Large_Silence_%', 'Overall_Silence_%'
                ]].copy()
                def_display.columns = [
                    'Rank', 'Agent', 'Total Calls', 'Short Calls', 'Short Silence %',
                    'Medium Calls', 'Medium Silence %', 'Large Calls', 'Large Silence %', 'Overall Silence %'
                ]
                st.dataframe(def_display, use_container_width=True, height=300)

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
                "⬇️ Download Call Data", data=buf_s,
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
                "⬇️ Download Performance Report", data=buf_p,
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
                "⬇️ Download Defaulter Report", data=buf_d,
                file_name=f"Defaulter_Report_{client_name.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_complete_def",
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 6: Email & Reports Portal ----
    st.markdown('<div class="step-card" style="border: 2px solid #4F46E5; background: #F5F4FF;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-title"><span class="step-badge" style="background:#7C3AED;">6</span>📧 Email Reports</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="step-subtitle">Configure SMTP, manage recipients, set scheduler time, and send reports.</p>', unsafe_allow_html=True)
    
    # Show current status
    smtp_config = st.session_state.get("smtp_config", DEFAULT_SMTP_CONFIG.copy())
    st.caption(
        f"⏰ Auto-reports: {'Enabled' if st.session_state.get('scheduler_enabled', True) else 'Disabled'} · "
        f"Time: {st.session_state.get('scheduler_time', '23:00')} · "
        f"Recipients: {len(st.session_state.mentor_emails)} · "
        f"SMTP: {smtp_config.get('host', 'Not set')}:{smtp_config.get('port', '')}"
    )
    
    if st.session_state.get("last_email_time"):
        st.caption(f"📨 Last report sent: {st.session_state['last_email_time']}")
    
    if st.button("📧 Open Email Portal", type="primary"):
        open_email_portal(client_name)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Pick a client and date range above, then click **Fetch Calls** to get started.")
