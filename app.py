import streamlit as st
import pandas as pd
import yaml
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai
from openai import OpenAI
import os
import io
import traceback
import re
from collections import Counter
import time
import base64
import json
import ast

# --- Conditional Imports for OCR and XAI SDK ---
try:
    from xai_sdk.client import Client
    from xai_sdk.chat import user, system
    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_SDK_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import pdf2image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Enhanced Agentic Analysis System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced Theme Definitions ---
themes = {
    "Blue Sky": {
        "primaryColor": "#00BFFF",
        "backgroundColor": "#E6F3FF",
        "secondaryBackgroundColor": "#B3D9FF",
        "textColor": "#003366",
        "accentColor": "#0080FF"
    },
    "Snow White": {
        "primaryColor": "#A0A0A0",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F8F8F8",
        "textColor": "#2C2C2C",
        "accentColor": "#D0D0D0"
    },
    # ... (other themes remain the same)
}

# --- State Initialization ---
if 'theme' not in st.session_state:
    st.session_state.theme = "Blue Sky"
if 'last_agent_output' not in st.session_state:
    st.session_state.last_agent_output = ""
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        "Gemini": os.getenv("GEMINI_API_KEY", ""),
        "OpenAI": os.getenv("OPENAI_API_KEY", ""),
        "Grok": os.getenv("XAI_API_KEY", "")
    }
if 'raw_combined' not in st.session_state:
    st.session_state.raw_combined = ""
if 'article2_input' not in st.session_state:
    st.session_state.article2_input = ""
if 'mind_map_relationships' not in st.session_state:
    st.session_state.mind_map_relationships = []
# ... (other state initializations remain the same)

# Apply theme with enhanced styling
current_theme = themes.get(st.session_state.theme, themes["Blue Sky"])
st.markdown(f"""
<style>
    .stApp {{
        background-color: {current_theme['backgroundColor']};
        color: {current_theme['textColor']};
    }}
    .coral-keyword {{
        color: #FF7F50; /* Coral color for auto-detected keywords */
        font-weight: bold;
    }}
    .orange-keyword {{
        background-color: #FFA500; /* Orange background for user-specified keywords */
        color: white;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }}
    .fancy-header {{
        background: linear-gradient(90deg, {current_theme['primaryColor']}, {current_theme['accentColor']});
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    /* ... (other styles remain the same) */
</style>
""", unsafe_allow_html=True)

# --- All function definitions (call_llm_api, ocr, etc.) remain the same ---
# ... (omitting unchanged functions for brevity)

# --- NEW/MODIFIED FUNCTIONS ---

def highlight_specific_keywords(text, keywords):
    """Highlights a user-provided list of keywords in a given text."""
    for keyword in keywords:
        # Use word boundaries to avoid highlighting parts of words
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        text = pattern.sub(f'<span class="orange-keyword">{keyword}</span>', text)
    return text

def create_comparison_prompt(doc1, doc2, keywords):
    """Creates a detailed prompt for comparing two documents based on keywords."""
    keywords_str = ", ".join(keywords)
    return f"""You are an expert analyst. Below are two documents and a list of specific keywords. Your task is to generate a comprehensive report that compares and contrasts these two documents, with a strong focus on how they discuss, define, or relate to these keywords.

Structure your report with a main summary, followed by clear headings for each keyword. Under each keyword, analyze:
- The context and sentiment in which it appears in Document 1.
- The context and sentiment in which it appears in Document 2.
- The similarities and differences in their treatment of the keyword.
- Any unique insights each document provides regarding the keyword.

Keywords to focus on: {keywords_str}

Document 1 (first 5000 chars):
{doc1[:5000]}

Document 2 (first 5000 chars):
{doc2[:5000]}

Generate the comparison report in markdown format.
"""

# --- Main Application Logic ---

# ... (Sidebar and Document Upload sections remain the same)

# --- Analysis Section ---
if st.session_state.raw_combined:
    # ... (Tabs for Summary, Word Cloud, Entities, Agents remain the same)
    pass

# --- Mind Map Section ---
st.header("🧠 Cross-Document Mind Map Generation")

# Use session state to keep the article 2 text persistent
st.session_state.article2_input = st.text_area(
    "Paste a second article to compare and find relationships",
    value=st.session_state.article2_input,
    height=200
)

if st.button("Generate Relationships") and st.session_state.article2_input and st.session_state.raw_combined:
    # ... (Relationship generation logic remains the same)
    pass

if st.session_state.mind_map_relationships:
    st.subheader("Interactive Mind Map")
    html_content = create_interactive_mindmap(st.session_state.mind_map_relationships)
    if html_content:
        st.components.v1.html(html_content, height=650)
    
    # --- NEW: EDITABLE RELATIONSHIPS ---
    st.subheader("Edit Relationships")
    # Format relationships for display in the text area
    rels_as_str = "\n".join([str(rel) for rel in st.session_state.mind_map_relationships])
    
    edited_rels_str = st.text_area(
        "Edit the relationships below (one Python tuple per line) and click 'Update Mind Map'.",
        value=rels_as_str,
        height=250
    )
    
    if st.button("Update Mind Map", use_container_width=True):
        new_rels = []
        try:
            # Parse the string back into a list of tuples
            for line in edited_rels_str.strip().split('\n'):
                if line.strip():
                    # ast.literal_eval is a safe way to parse Python literals
                    parsed_tuple = ast.literal_eval(line.strip())
                    if isinstance(parsed_tuple, tuple) and len(parsed_tuple) == 3:
                        new_rels.append(parsed_tuple)
                    else:
                        st.warning(f"Skipping malformed line: {line}")
            
            st.session_state.mind_map_relationships = new_rels
            st.success("Mind map updated successfully!")
            # Rerun to display the updated map immediately
            st.rerun()
            
        except Exception as e:
            st.error(f"Error parsing relationships: {e}. Please ensure each line is a valid Python tuple, e.g., ('Source', 'Target', 'Relation').")

# --- NEW: KEYWORD-DRIVEN COMPARISON SECTION ---
st.header("🔍 Keyword-Driven Comparison")

if st.session_state.raw_combined and st.session_state.article2_input:
    user_keywords_str = st.text_area(
        "Paste comma-separated keywords to focus the comparison (e.g., AI, market trend, data privacy).",
        height=100
    )

    if st.button("Analyze with Keywords", use_container_width=True):
        if user_keywords_str:
            # Parse keywords
            keywords = [k.strip() for k in user_keywords_str.split(',') if k.strip()]
            st.session_state.user_keywords = keywords
            
            with st.spinner("Analyzing documents with your keywords..."):
                # Highlight documents
                st.session_state.highlighted_doc1 = highlight_specific_keywords(st.session_state.raw_combined, keywords)
                st.session_state.highlighted_doc2 = highlight_specific_keywords(st.session_state.article2_input, keywords)
                
                # Generate comparison report
                comparison_prompt = create_comparison_prompt(st.session_state.raw_combined, st.session_state.article2_input, keywords)
                
                # Using Gemini as default for this task, can be changed to a selectbox
                report = call_llm_api(
                    "Gemini",
                    st.session_state.api_keys.get("Gemini"),
                    "gemini-1.5-flash",
                    comparison_prompt
                )
                if report:
                    st.session_state.comparison_report = report
                else:
                    st.error("Failed to generate the comparison report.")
        else:
            st.warning("Please enter at least one keyword.")

    # Display the results if they exist in the session state
    if 'comparison_report' in st.session_state:
        st.subheader("Highlighted Documents")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Document 1 with Keywords", expanded=True):
                st.markdown(st.session_state.highlighted_doc1, unsafe_allow_html=True)
        with col2:
            with st.expander("Document 2 with Keywords", expanded=True):
                st.markdown(st.session_state.highlighted_doc2, unsafe_allow_html=True)
        
        st.subheader("Comparative Analysis Report")
        st.markdown(st.session_state.comparison_report)
        st.markdown(
            create_download_link(st.session_state.comparison_report, "keyword_comparison_report.md"),
            unsafe_allow_html=True
        )

else:
    st.info("Please ensure both Document 1 (from the top section) and the second article are loaded to use this feature.")

st.markdown("---")
st.info("💡 Application enhanced with editable mind maps and keyword-driven comparative analysis.")
