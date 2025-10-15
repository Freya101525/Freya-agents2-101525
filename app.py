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
    "Sparkling Stars": {
        "primaryColor": "#FFD700",
        "backgroundColor": "#0A0E27",
        "secondaryBackgroundColor": "#1A1F3A",
        "textColor": "#E0E0E0",
        "accentColor": "#FFE55C"
    },
    "Alps Forest": {
        "primaryColor": "#228B22",
        "backgroundColor": "#F0FFF0",
        "secondaryBackgroundColor": "#D4EDD4",
        "textColor": "#1B4D1B",
        "accentColor": "#32CD32"
    },
    "Flora Garden": {
        "primaryColor": "#FF69B4",
        "backgroundColor": "#FFF0F5",
        "secondaryBackgroundColor": "#FFE4E9",
        "textColor": "#8B008B",
        "accentColor": "#FF1493"
    },
    "Fresh Air": {
        "primaryColor": "#00CED1",
        "backgroundColor": "#F0FFFF",
        "secondaryBackgroundColor": "#E0F7F7",
        "textColor": "#006B6F",
        "accentColor": "#40E0D0"
    },
    "Deep Ocean": {
        "primaryColor": "#00FFFF",
        "backgroundColor": "#001F3F",
        "secondaryBackgroundColor": "#003366",
        "textColor": "#B0E0E6",
        "accentColor": "#1E90FF"
    },
    "Ferrari Sportscar": {
        "primaryColor": "#FF2800",
        "backgroundColor": "#0D0D0D",
        "secondaryBackgroundColor": "#1A1A1A",
        "textColor": "#FFFFFF",
        "accentColor": "#FF6347"
    },
    "Fendi Casa Luxury": {
        "primaryColor": "#C9A87C",
        "backgroundColor": "#FBF8F3",
        "secondaryBackgroundColor": "#F5EFE6",
        "textColor": "#4A3F35",
        "accentColor": "#D4AF77"
    }
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
if 'combined_markdown' not in st.session_state:
    st.session_state.combined_markdown = ""
if 'raw_combined' not in st.session_state:
    st.session_state.raw_combined = ""
if 'article2_markdown' not in st.session_state:
    st.session_state.article2_markdown = ""
if 'mind_map_relationships' not in st.session_state:
    st.session_state.mind_map_relationships = []
if 'entity_categories' not in st.session_state:
    st.session_state.entity_categories = [
        "Person", "Organization", "Location", "Technology", 
        "Concept", "Product", "Event", "Date", "Metric", "Other"
    ]
if 'current_agent_index' not in st.session_state:
    st.session_state.current_agent_index = 0

# Apply theme with enhanced styling
current_theme = themes.get(st.session_state.theme, themes["Blue Sky"])
st.markdown(f"""
<style>
    .stApp {{
        background-color: {current_theme['backgroundColor']};
        color: {current_theme['textColor']};
    }}
    .coral-keyword {{
        color: #FF7F50;
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
    .progress-container {{
        background-color: {current_theme['secondaryBackgroundColor']};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }}
    .export-button {{
        background-color: {current_theme['primaryColor']};
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)


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



# --- Fancy Progress Indicator ---
def show_fancy_progress(message, duration=2):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.markdown(f"**{message}** {'.' * ((i // 25) % 4)}")
        time.sleep(duration / 100)
    progress_bar.empty()
    status_text.empty()

# --- Export Functions ---
def create_download_link(content, filename, file_type="text/plain"):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:{file_type};base64,{b64}" download="{filename}" class="export-button">⬇️ Download {filename}</a>'

def export_entities_csv(entities_md):
    lines = entities_md.strip().split('\n')
    csv_content = ""
    for line in lines:
        if '|' in line and not line.startswith('|---'):
            csv_content += ','.join([cell.strip() for cell in line.split('|')[1:-1]]) + '\n'
    return csv_content

# --- API Call Functions ---
def call_llm_api(provider, api_key, model_name, prompt, system_prompt="You are a helpful AI assistant."):
    if not api_key:
        st.error(f"API key for {provider} is not set. Please add it in the sidebar.")
        return None
    
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        
        elif provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        
        elif provider == "Grok":
            if not XAI_SDK_AVAILABLE:
                st.error("XAI SDK (Grok) is not installed. Please install it to use this provider.")
                return None
            
            # --- UPDATED CODE BLOCK ---
            # Aligned with the official sample code, including the recommended timeout.
            client = Client(api_key=api_key, timeout=3600)
            chat = client.chat.create(model=model_name)
            chat.append(system(system_prompt))
            chat.append(user(prompt))
            response = chat.sample()
            return response.content
            # --- END OF UPDATED CODE BLOCK ---
            
    except Exception as e:
        st.error(f"An error occurred with the {provider} API: {e}")
        traceback.print_exc()
        return None

# --- OCR Functions ---
def ocr_pdf_pytesseract(pdf_file, pages):
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        images = pdf2image.convert_from_bytes(pdf_bytes, first_page=pages[0], last_page=pages[-1])
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except Exception as e:
        st.error(f"Pytesseract OCR Error: {e}")
        return None

def ocr_pdf_easyocr(pdf_file, pages):
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        reader = easyocr.Reader(['en'])
        images = pdf2image.convert_from_bytes(pdf_bytes, first_page=pages[0], last_page=pages[-1])
        text = ""
        for img in images:
            result = reader.readtext(img, detail=0, paragraph=True)
            text += "\n".join(result) + "\n"
        return text
    except Exception as e:
        st.error(f"EasyOCR Error: {e}")
        return None

# --- Text Processing Functions ---
def extract_keywords(text, top_n=50):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 'could', 'should', 'also', 'however', 'therefore'}
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(top_n)]

def highlight_keywords(text, keywords):
    for keyword in keywords:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        text = pattern.sub(f'<span class="coral-keyword">{keyword}</span>', text)
    return text

def create_entities_prompt(text, categories):
    categories_str = ", ".join(categories)
    return f"""Extract up to 100 unique and most relevant entities from the following text.
Return them as a markdown table with columns: Entity | Type | Description

Available types: {categories_str}

Text (first 8000 characters):
{text[:8000]}

Return ONLY the markdown table, without any introductory or concluding text.
"""

# --- Mind Map Functions ---
def create_interactive_mindmap(relationships):
    if not relationships:
        st.warning("No relationships found to generate a mind map.")
        return ""
    
    G = nx.DiGraph()
    for source, target, relation in relationships:
        G.add_edge(str(source), str(target), title=str(relation), label=str(relation)[:20])
    
    net = Network(height="600px", width="100%", notebook=True, directed=True, bgcolor=current_theme['backgroundColor'], font_color=current_theme['textColor'])
    net.from_nx(G)
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 250
            },
            "minVelocity": 0.75
        }
    }
    """)
    try:
        net.save_graph("mindmap.html")
        with open("mindmap.html", 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Failed to generate mind map HTML: {e}")
        return ""

@st.cache_data
def load_agents():
    try:
        with open("agents.yaml", 'r') as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        st.error("agents.yaml not found. Please create this file in your project directory.")
        return {"agents": []}
    except Exception as e:
        st.error(f"Error loading agents.yaml: {e}")
        return {"agents": []}

# --- Sidebar UI ---
with st.sidebar:
    st.markdown('<div class="fancy-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    st.subheader("🎨 Theme Selection")
    st.selectbox("Choose Your Theme", options=list(themes.keys()), key="theme")
    
    st.divider()
    st.subheader("🔑 API Configuration")
    
    with st.expander("Gemini API", expanded=False):
        st.session_state.api_keys["Gemini"] = st.text_input("Gemini API Key", type="password", value=st.session_state.api_keys.get("Gemini"), key="gemini_key")
    
    with st.expander("OpenAI API", expanded=False):
        st.session_state.api_keys["OpenAI"] = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_keys.get("OpenAI"), key="openai_key")
    
    with st.expander("Grok API", expanded=False):
        st.session_state.api_keys["Grok"] = st.text_input("XAI (Grok) API Key", type="password", value=st.session_state.api_keys.get("Grok"), key="grok_key")
    
    st.divider()
    st.subheader("🏷️ Entity Categories")
    categories_text = st.text_area("Edit categories (one per line)", "\n".join(st.session_state.entity_categories), height=200)
    if st.button("Update Categories"):
        st.session_state.entity_categories = [c.strip() for c in categories_text.split('\n') if c.strip()]
        st.success("Categories updated!")

# --- Main Page ---
st.markdown('<div class="fancy-header">✨ Enhanced Agentic Analysis System</div>', unsafe_allow_html=True)

# --- Document Upload Section ---
st.header("📄 Document Processing")

num_docs = st.number_input("How many documents to process?", min_value=1, max_value=10, value=1)
documents = [None] * num_docs

for i in range(num_docs):
    with st.expander(f"📑 Document {i+1}", expanded=(i==0)):
        input_method = st.radio(f"Input method for Doc {i+1}", ['Upload', 'Paste'], key=f"method_{i}", horizontal=True)
        
        if input_method == 'Upload':
            uploaded = st.file_uploader(f"Upload file (txt, md, pdf)", type=["txt", "md", "pdf"], key=f"upload_{i}")
            if uploaded:
                if uploaded.type == "application/pdf":
                    st.subheader("PDF OCR Configuration")
                    
                    ocr_options = []
                    if PYTESSERACT_AVAILABLE: ocr_options.append("pytesseract")
                    if EASYOCR_AVAILABLE: ocr_options.append("easyocr")
                    if not ocr_options:
                        st.warning("No OCR libraries found. Please install `pytesseract` or `easyocr`.")
                    else:
                        ocr_method = st.selectbox("OCR Method", ocr_options, key=f"ocr_{i}")
                        page_range_str = st.text_input("Pages to OCR (e.g., 1-5 or 1,3,5)", "1-5", key=f"pages_{i}")
                        
                        if st.button(f"🔄 Process PDF {i+1}", key=f"process_{i}"):
                            with st.spinner("🔄 Processing PDF..."):
                                try:
                                    if '-' in page_range_str:
                                        start, end = map(int, page_range_str.split('-'))
                                        pages = list(range(start, end + 1))
                                    else:
                                        pages = [int(p.strip()) for p in page_range_str.split(',')]
                                except ValueError:
                                    st.error("Invalid page range format. Use '1-5' or '1,3,5'.")
                                else:
                                    text = None
                                    if ocr_method == "pytesseract":
                                        text = ocr_pdf_pytesseract(uploaded, pages)
                                    elif ocr_method == "easyocr":
                                        text = ocr_pdf_easyocr(uploaded, pages)
                                    
                                    if text:
                                        documents[i] = text
                                        st.success(f"✅ PDF {i+1} processed successfully!")
                                        st.text_area(f"Preview of Doc {i+1}", text[:500] + "...", height=150)
                else:
                    documents[i] = uploaded.read().decode('utf-8')
                    st.success(f"✅ Document {i+1} loaded!")
        else:
            pasted = st.text_area(f"Paste text for Doc {i+1}", height=200, key=f"paste_{i}")
            if pasted:
                documents[i] = pasted

if st.button("🔗 Combine All Documents", use_container_width=True) and any(documents):
    with st.spinner("Combining documents..."):
        valid_docs = [doc for doc in documents if doc]
        combined_text = "\n\n---\n\n".join([f"## Document {i+1}\n\n{doc}" for i, doc in enumerate(valid_docs)])
        keywords = extract_keywords(combined_text)
        
        st.session_state.raw_combined = combined_text
        st.session_state.combined_markdown = f"# Combined Document Analysis\n\n{highlight_keywords(combined_text, keywords)}"
        st.success("✅ Documents combined successfully!")

# --- Analysis Section ---
if st.session_state.raw_combined:
    st.header("🔬 Document Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Combined View", "📝 Summary", "☁️ Word Cloud", "🏷️ Entities", "🤖 Agents"])
    
    with tab1:
        st.markdown(create_download_link(st.session_state.raw_combined, "combined_document.md"), unsafe_allow_html=True)
        st.markdown(st.session_state.combined_markdown, unsafe_allow_html=True)

    with tab2:
        st.subheader("Comprehensive Summary")
        col1, col2 = st.columns(2)
        summary_provider = col1.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], key="sum_prov")
        summary_model_options = {
            "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite"], 
            "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano"], 
            "Grok": ["grok-4-fast-reasoning", "grok-3-mini"]
        }
        summary_model = col2.selectbox("Model", summary_model_options.get(summary_provider, []), key="sum_model")
        
        if st.button("📄 Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = call_llm_api(summary_provider, st.session_state.api_keys.get(summary_provider), summary_model, f"Create a comprehensive markdown summary with key points, themes, and insights:\n\n{st.session_state.raw_combined[:15000]}")
                if summary: st.session_state.summary = summary
        
        if 'summary' in st.session_state:
            st.markdown(st.session_state.summary)
            st.markdown(create_download_link(st.session_state.summary, "summary.md"), unsafe_allow_html=True)

    with tab3:
        st.subheader("Word Cloud Visualization")
        if st.button("🎨 Generate Word Cloud", use_container_width=True):
            with st.spinner("Creating word cloud..."):
                text = re.sub('<[^<]+?>', '', st.session_state.raw_combined)
                wordcloud = WordCloud(width=1200, height=600, background_color=current_theme['backgroundColor'], colormap='viridis', max_words=150).generate(text)
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                st.download_button("⬇️ Download Word Cloud", buf, "wordcloud.png", "image/png", use_container_width=True)

    with tab4:
        st.subheader("Entity Extraction")
        if st.button("🔍 Extract Entities", use_container_width=True):
            with st.spinner("Extracting entities..."):
                entity_prompt = create_entities_prompt(st.session_state.raw_combined, st.session_state.entity_categories)
                entities = call_llm_api(summary_provider, st.session_state.api_keys.get(summary_provider), summary_model, entity_prompt)
                if entities: st.session_state.entities = entities
        
        if 'entities' in st.session_state:
            st.markdown(st.session_state.entities)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(create_download_link(st.session_state.entities, "entities.md"), unsafe_allow_html=True)
            with col2:
                csv_entities = export_entities_csv(st.session_state.entities)
                st.markdown(create_download_link(csv_entities, "entities.csv", "text/csv"), unsafe_allow_html=True)
      # --- AGENT EXECUTION TAB (CORRECTED) ---
    with tab5:
        st.subheader("🤖 Agentic Execution")
        agents = load_agents().get('agents', [])
        
        model_options = {
            "Gemini": ["gemini-1.5-flash", "gemini-pro"], 
            "OpenAI": ["gpt-4o", "gpt-3.5-turbo"], 
            "Grok": ["grok-4", "grok-1.5-flash"]
        }
        
        if agents:
            execution_mode = st.radio(
                "Select Execution Mode",
                ["Single Agent", "Sequential Pipeline"],
                horizontal=True,
                help="Choose 'Single Agent' to run one specific agent, or 'Sequential Pipeline' to run a series of agents in order."
            )
            st.divider()

            # --- SINGLE AGENT MODE ---
            if execution_mode == "Single Agent":
                st.markdown("#### Run a Specific Agent")
                
                agent_names = [agent.get('name', f'Agent {i+1}') for i, agent in enumerate(agents)]
                selected_agent_name = st.selectbox("Choose an agent to run", agent_names)
                
                selected_agent = next((agent for agent in agents if agent.get('name') == selected_agent_name), None)
                
                if selected_agent:
                    st.info(f"**Category:** {selected_agent.get('category', 'General')} | **Description:** {selected_agent.get('description', 'No description')}")
                    
                    col1, col2 = st.columns(2)
                    agent_provider = col1.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], key="single_agent_prov")
                    agent_model = col2.selectbox("Model", model_options.get(agent_provider, []), key="single_agent_model")
                    
                    prompt = st.text_area("Agent Prompt", selected_agent.get('prompt', ''), height=150, key="single_agent_prompt")
                    use_previous = st.checkbox("Use previous agent's output as input", value=True, key="single_use_previous")
                    
                    if st.button(f"▶️ Execute '{selected_agent_name}'", use_container_width=True):
                        with st.spinner(f"Agent '{selected_agent_name}' is working..."):
                            input_data = st.session_state.last_agent_output if (use_previous and st.session_state.last_agent_output) else st.session_state.raw_combined
                            
                            result = call_llm_api(
                                agent_provider, 
                                st.session_state.api_keys.get(agent_provider), 
                                agent_model,
                                f"{prompt}\n\nData:\n{input_data[:10000]}"
                            )
                            
                            if result:
                                st.session_state.last_agent_output = result
                                st.success("✅ Agent executed successfully!")
                                st.markdown("### Agent Output")
                                st.markdown(result)
                                st.download_button("⬇️ Download Output", result, f"{selected_agent_name}_output.md", use_container_width=True)
                            else:
                                st.error("Agent execution failed or returned no output.")
                
            # --- SEQUENTIAL PIPELINE MODE (Corrected Block) ---
            elif execution_mode == "Sequential Pipeline":
                st.markdown("#### Run a Series of Agents")
                num_agents = st.slider("Number of agents to execute in sequence", 1, min(len(agents), 31), 5)
                
                if st.session_state.current_agent_index < num_agents:
                    agent = agents[st.session_state.current_agent_index]
                    st.subheader(f"🎯 Pipeline Step {st.session_state.current_agent_index + 1}/{num_agents}: {agent.get('name', 'Unnamed Agent')}")
                    st.info(f"**Category:** {agent.get('category', 'General')} | **Description:** {agent.get('description', 'No description')}")
                    
                    col1, col2 = st.columns(2)
                    agent_provider = col1.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], key="pipe_agent_prov")
                    agent_model = col2.selectbox("Model", model_options.get(agent_provider, []), key="pipe_agent_model")
                    
                    prompt = st.text_area("Agent Prompt", agent.get('prompt', ''), height=150, key="pipe_agent_prompt")
                    use_previous = st.checkbox("Use previous agent's output as input", value=True, key="pipe_use_previous")
                    
                    if st.button("▶️ Execute and Advance to Next Step", use_container_width=True):
                        with st.spinner(f"Agent '{agent.get('name')}' is working..."):
                            input_data = st.session_state.last_agent_output if (use_previous and st.session_state.last_agent_output) else st.session_state.raw_combined
                            
                            # --- THIS IS THE FULLY CORRECTED FUNCTION CALL ---
                            # No typos and a correct, matching pair of parentheses.
                            result = call_llm_api(
                                agent_provider, 
                                st.session_state.api_keys.get(agent_provider), 
                                agent_model,
                                f"{prompt}\n\nData:\n{input_data[:10000]}"
                            )
                            # --- END OF FIX ---

                            if result:
                                st.session_state.last_agent_output = result
                                st.session_state[f'agent_{st.session_state.current_agent_index}_output'] = result
                                st.session_state.current_agent_index += 1
                                st.rerun()
                else:
                    st.success("✅ All pipeline steps executed successfully!")
                
                if st.button("🔄 Reset Pipeline", use_container_width=True):
                    st.session_state.current_agent_index = 0
                    st.session_state.last_agent_output = ""
                    for i in range(len(agents)):
                        st.session_state.pop(f'agent_{i}_output', None)
                    st.rerun()

                with st.expander("📊 View Pipeline Outputs"):
                    # ... (Unchanged)
                    pass
        else:
            st.warning("⚠️ No agents found. Please create an `agents.yaml` file.")


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
                    "gemini-2.5-flash-lite",
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
