from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0b0b;
            --panel: #141414;
            --panel-2: #1d1d1d;
            --border: #2f2f2f;
            --accent: #ff7a00;
            --accent-2: #ff9a3d;
            --text: #f5f5f5;
            --muted: #b7b7b7;
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255,122,0,0.18), transparent 30%),
                linear-gradient(180deg, #090909 0%, #111111 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #120d08 0%, #0c0c0c 100%);
            border-right: 1px solid var(--border);
        }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .hero {
            padding: 1.5rem;
            border: 1px solid rgba(255,122,0,0.25);
            background: linear-gradient(135deg, rgba(255,122,0,0.18), rgba(20,20,20,0.92));
            border-radius: 20px;
            margin-bottom: 1rem;
            box-shadow: 0 18px 48px rgba(0,0,0,0.35);
        }
        .card {
            background: linear-gradient(180deg, rgba(22,22,22,0.96), rgba(14,14,14,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-2);
        }
        .metric { font-size: 1.8rem; font-weight: 700; margin-top: 0.35rem; }
        div[data-testid="stMetric"] {
            background: rgba(18,18,18,0.9);
            border: 1px solid rgba(255,122,0,0.15);
            padding: 0.75rem;
            border-radius: 16px;
        }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(180deg, #ff8b1f 0%, #ff6a00 100%);
            color: #111111;
            border: none;
            font-weight: 700;
            border-radius: 999px;
        }
        .stTextArea textarea, .stTextInput input, .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {
            background: rgba(18,18,18,0.95);
            color: var(--text);
        }
        .schema-preview-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }
        .schema-preview-table th {
            background: rgba(255,122,0,0.15);
            color: #ff9a3d;
            text-align: left;
            padding: 6px 10px;
            border-bottom: 1px solid rgba(255,122,0,0.25);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .schema-preview-table td {
            padding: 6px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            color: #e0e0e0;
        }
        .schema-preview-table tr:last-child td { border-bottom: none; }
        .badge-req {
            background: rgba(255,122,0,0.25);
            color: #ff9a3d;
            border-radius: 4px;
            padding: 1px 6px;
            font-size: 0.7rem;
        }
        .badge-opt {
            background: rgba(255,255,255,0.06);
            color: #888;
            border-radius: 4px;
            padding: 1px 6px;
            font-size: 0.7rem;
        }
        .badge-gated {
            background: rgba(255,60,60,0.2);
            color: #ff7070;
            border-radius: 4px;
            padding: 1px 6px;
            font-size: 0.7rem;
            margin-left: 4px;
        }
        .dataset-card {
            background: rgba(18,18,18,0.8);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
        }
        pre, code { border-radius: 14px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
