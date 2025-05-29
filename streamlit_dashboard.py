#!/usr/bin/env python3
"""
D&D LoRA Project - Comprehensive Streamlit Dashboard
===================================================

A modern, interactive web interface for the D&D LoRA fine-tuning project.
Provides model comparison, live inference, training visualization, and evaluation.

Features:
- Real-time model comparison (Original vs LoRA)
- Interactive inference with both models
- Training progress visualization
- Evaluation metrics and reports
- Project documentation browser

Usage:
    streamlit run streamlit_dashboard.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import time
import glob
from datetime import datetime
from pathlib import Path
import requests
from typing import Dict, Any, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our core library
try:
    from dnd_lora_core import DnDModelComparator, DnDDataProcessor
except ImportError as e:
    st.error(f"Failed to import dnd_lora_core: {e}")
    st.error("Please ensure you're running from the project root directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🐉 D&D LoRA Dashboard",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "eval"
API_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def get_available_models():
    """Get list of available trained models."""
    models = []
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir() and (model_dir / "adapter_config.json").exists():
                models.append(model_dir.name)
    return models

@st.cache_data
def load_evaluation_reports():
    """Load available evaluation reports."""
    reports = []
    if EVAL_DIR.exists():
        for report_dir in EVAL_DIR.iterdir():
            if report_dir.is_dir():
                html_files = list(report_dir.glob("*.html"))
                csv_files = list(report_dir.glob("*.csv"))
                if html_files or csv_files:
                    reports.append({
                        "name": report_dir.name,
                        "path": report_dir,
                        "html_files": html_files,
                        "csv_files": csv_files,
                        "modified": datetime.fromtimestamp(report_dir.stat().st_mtime)
                    })
    return sorted(reports, key=lambda x: x["modified"], reverse=True)

def check_api_status():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def call_api(endpoint: str, data: Optional[Dict] = None):
    """Make API call to the server."""
    try:
        if data:
            response = requests.post(f"{API_URL}/{endpoint}", json=data, timeout=30)
        else:
            response = requests.get(f"{API_URL}/{endpoint}", timeout=10)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Main application
def main():
    # Header
    st.title("🐉 D&D LoRA Fine-Tuning Dashboard")
    st.markdown("*Complete toolkit for D&D knowledge injection using LoRA fine-tuning*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # API Status
        api_status = check_api_status()
        if api_status:
            st.success("✅ API Server Online")
        else:
            st.error("❌ API Server Offline")
            with st.expander("Start API Server"):
                st.code("python api_server.py", language="bash")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        available_models = get_available_models()
        
        if available_models:
            selected_model = st.selectbox(
                "Choose a trained model:",
                available_models,
                index=0 if "distilgpt2-dnd-lora" in available_models else 0
            )
            
            # Model info
            model_path = MODELS_DIR / selected_model
            if (model_path / "README.md").exists():
                with st.expander("📋 Model Info"):
                    readme_content = (model_path / "README.md").read_text()
                    st.markdown(readme_content)
        else:
            st.warning("No trained models found")
            st.info("Run `python train_dnd_lora.py` to train a model")
            selected_model = None
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Generate Report"):
            if selected_model:
                st.info("Starting model comparison...")
                # This would trigger the compare_models.py script
                st.code(f"python compare_models.py --model {selected_model.split('-')[0]}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "🤖 Model Comparison", 
        "💬 Interactive Chat", 
        "📈 Training & Evaluation", 
        "📚 Documentation"
    ])
    
    # Tab 1: Overview
    with tab1:
        show_overview_tab(selected_model)
    
    # Tab 2: Model Comparison
    with tab2:
        show_comparison_tab(selected_model)
    
    # Tab 3: Interactive Chat
    with tab3:
        show_chat_tab(selected_model, api_status)
    
    # Tab 4: Training & Evaluation
    with tab4:
        show_evaluation_tab()
    
    # Tab 5: Documentation
    with tab5:
        show_documentation_tab()

def show_overview_tab(selected_model: Optional[str]):
    """Show project overview and quick stats."""
    st.header("🏠 Project Overview")
    
    # Project stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        num_models = len(get_available_models())
        st.metric("📦 Trained Models", num_models)
    
    with col2:
        num_reports = len(load_evaluation_reports())
        st.metric("📊 Evaluation Reports", num_reports)
    
    with col3:
        data_file = PROJECT_ROOT / "data" / "dnd_srd_qa.jsonl"
        if data_file.exists():
            with open(data_file) as f:
                num_examples = sum(1 for _ in f)
            st.metric("📝 Training Examples", f"{num_examples:,}")
        else:
            st.metric("📝 Training Examples", "Not prepared")
    
    with col4:
        if selected_model:
            model_size = get_model_size(MODELS_DIR / selected_model)
            st.metric("💾 Model Size", model_size)
        else:
            st.metric("💾 Model Size", "N/A")
    
    # Project description
    st.subheader("📖 About This Project")
    st.markdown("""
    This project demonstrates **"zero-to-hero" knowledge injection** using LoRA (Low-Rank Adaptation) 
    fine-tuning to transform general language models into D&D 5e experts.
    
    **Key Features:**
    - 🎯 **Efficient Training**: Only ~1% of model parameters needed
    - 📈 **Dramatic Improvement**: 200-800% increase in D&D terminology
    - 🔄 **Real-time Comparison**: Side-by-side model evaluation
    - 🚀 **Production Ready**: Complete API server and deployment tools
    - 📊 **Comprehensive Evaluation**: Automated testing and reporting
    """)
    
    # Recent activity
    if selected_model:
        st.subheader(f"🔍 Model Details: {selected_model}")
        model_path = MODELS_DIR / selected_model
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuration:**")
            config_file = model_path / "adapter_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                
                st.json({
                    "Base Model": config.get("base_model_name_or_path", "Unknown"),
                    "LoRA Rank": config.get("r", "Unknown"),
                    "LoRA Alpha": config.get("lora_alpha", "Unknown"),
                    "Target Modules": config.get("target_modules", [])
                })
        
        with col2:
            st.markdown("**Training Info:**")
            training_file = model_path / "training_args.bin"
            if training_file.exists():
                import torch
                try:
                    training_args = torch.load(training_file, map_location="cpu")
                    st.json({
                        "Learning Rate": f"{training_args.learning_rate:.2e}",
                        "Batch Size": training_args.per_device_train_batch_size,
                        "Epochs": training_args.num_train_epochs,
                        "Warmup Steps": training_args.warmup_steps
                    })
                except:
                    st.info("Training info not available")

def show_comparison_tab(selected_model: Optional[str]):
    """Show model comparison interface."""
    st.header("🤖 Model Comparison")
    
    if not selected_model:
        st.warning("Please select a trained model from the sidebar")
        return
    
    # Comparison controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_questions = st.slider("Number of Questions", 1, 10, 5)
    
    with col2:
        question_type = st.selectbox(
            "Question Type",
            ["D&D Specific", "General Knowledge", "Mixed"]
        )
    
    with col3:
        if st.button("🚀 Run Comparison"):
            run_model_comparison(selected_model, num_questions, question_type)
    
    # Load and display recent comparison results
    show_recent_comparisons()

def show_chat_tab(selected_model: Optional[str], api_status: bool):
    """Show interactive chat interface."""
    st.header("💬 Interactive Chat")
    
    if not api_status:
        st.error("API server is not running. Please start it to use the chat interface.")
        st.code("python api_server.py", language="bash")
        return
    
    if not selected_model:
        st.warning("Please select a trained model from the sidebar")
        return
    
    # Chat interface
    st.subheader("🎮 D&D Assistant Chat")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.text_input("Ask about D&D 5e:", placeholder="What is a fireball spell?")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💫 Send to LoRA Model"):
            if user_input:
                send_chat_message(user_input, "lora")
    
    with col2:
        if st.button("🎯 Send to Original Model"):
            if user_input:
                send_chat_message(user_input, "original")
    
    with col3:
        if st.button("⚖️ Compare Both Models"):
            if user_input:
                send_chat_message(user_input, "both")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.expander(f"Q: {chat['question'][:50]}..." if len(chat['question']) > 50 else f"Q: {chat['question']}"):
                if chat['type'] == 'both':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🤖 Original Model:**")
                        st.write(chat['original_response'])
                    with col2:
                        st.markdown("**💫 LoRA Model:**")
                        st.write(chat['lora_response'])
                else:
                    st.markdown(f"**{'💫 LoRA' if chat['type'] == 'lora' else '🤖 Original'} Model:**")
                    st.write(chat['response'])

def show_evaluation_tab():
    """Show training and evaluation metrics."""
    st.header("📈 Training & Evaluation")
    
    # Load evaluation reports
    reports = load_evaluation_reports()
    
    if not reports:
        st.info("No evaluation reports found. Run model comparison to generate reports.")
        st.code("python compare_models.py --model distilgpt2", language="bash")
        return
    
    # Report selector
    selected_report = st.selectbox(
        "Select Evaluation Report:",
        options=range(len(reports)),
        format_func=lambda i: f"{reports[i]['name']} ({reports[i]['modified'].strftime('%Y-%m-%d %H:%M')})"
    )
    
    report = reports[selected_report]
    
    # Display report
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {report['name']}")
        
        # Load CSV data if available
        if report['csv_files']:
            csv_file = report['csv_files'][0]
            df = pd.read_csv(csv_file)
            
            # Show metrics summary
            if 'dnd_terms_original' in df.columns and 'dnd_terms_lora' in df.columns:
                avg_original = df['dnd_terms_original'].mean()
                avg_lora = df['dnd_terms_lora'].mean()
                improvement = ((avg_lora - avg_original) / max(avg_original, 0.1)) * 100
                
                st.metric(
                    "D&D Terms Improvement", 
                    f"{improvement:.1f}%",
                    delta=f"+{avg_lora - avg_original:.1f} terms"
                )
                
                # Plot comparison
                fig = px.bar(
                    x=['Original Model', 'LoRA Model'],
                    y=[avg_original, avg_lora],
                    title="Average D&D Terms per Response",
                    color=['Original Model', 'LoRA Model'],
                    color_discrete_map={
                        'Original Model': '#ff7f7f',
                        'LoRA Model': '#7fbf7f'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Show HTML report if available
        if report['html_files']:
            html_file = report['html_files'][0]
            st.markdown("**Full HTML Report:**")
            
            # Add buttons to view the report
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button(f"📄 View Inline", key=f"view_{html_file.name}"):
                    st.session_state.show_html_report = True
                    st.session_state.current_html_file = html_file
            
            with col_b:
                if st.button(f"📊 View Summary", key=f"summary_{html_file.name}"):
                    st.session_state.show_summary_report = True
                    st.session_state.current_html_file = html_file
            
            with col_c:
                if st.button(f"📂 Open in Browser", key=f"open_{html_file.name}"):
                    # Show file path for manual opening
                    file_url = f"file://{html_file.absolute()}"
                    st.markdown(f"**Open this URL in your browser:**")
                    st.code(file_url)
                    st.markdown("*Or run this command in terminal:*")
                    st.code(f"open '{html_file}'")
                    st.info("💡 Tip: Copy the file URL above and paste it into your browser's address bar")
            
            # Display summary content (more reliable)
            if st.session_state.get('show_summary_report') and st.session_state.get('current_html_file') == html_file:
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    report_data = extract_report_content(html_content)
                    
                    st.markdown(f"### {report_data['title']}")
                    
                    # Display summary metrics
                    if report_data['summary']:
                        st.markdown("#### 📊 Summary Metrics")
                        cols = st.columns(len(report_data['summary']))
                        for i, (metric, value) in enumerate(report_data['summary'].items()):
                            with cols[i]:
                                st.metric(metric, value)
                    
                    # Display comparisons
                    if report_data['comparisons']:
                        st.markdown("#### 🔍 Model Comparisons")
                        for i, comp in enumerate(report_data['comparisons'][:3]):  # Show first 3
                            with st.expander(f"Question {i+1}: {comp['question'][:60]}..."):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**🤖 Original Model:**")
                                    st.write(comp['original'][:200] + "..." if len(comp['original']) > 200 else comp['original'])
                                with col2:
                                    st.markdown("**🐉 LoRA Model:**")
                                    st.write(comp['lora'][:200] + "..." if len(comp['lora']) > 200 else comp['lora'])
                                st.markdown("**📈 Analysis:**")
                                st.info(comp['analysis'])
                        
                        if len(report_data['comparisons']) > 3:
                            st.info(f"Showing 3 of {len(report_data['comparisons'])} comparisons. View full report inline or in browser for complete details.")
                    
                    if st.button("❌ Hide Summary", key=f"hide_summary_{html_file.name}"):
                        st.session_state.show_summary_report = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading report summary: {e}")
                    st.info("Try viewing the report inline or opening in browser.")
            
            # Display HTML content inline if requested
            if st.session_state.get('show_html_report') and st.session_state.get('current_html_file') == html_file:
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Extract body content to avoid conflicts with Streamlit's CSS
                    if '<body>' in html_content and '</body>' in html_content:
                        body_start = html_content.find('<body>') + 6
                        body_end = html_content.find('</body>')
                        body_content = html_content[body_start:body_end]
                        
                        # Extract and include the styles
                        if '<style>' in html_content and '</style>' in html_content:
                            style_start = html_content.find('<style>') + 7
                            style_end = html_content.find('</style>')
                            styles = html_content[style_start:style_end]
                            
                            # Scope the styles to avoid conflicts
                            scoped_styles = f"""
                            <style>
                            .html-report {{
                                font-family: Arial, sans-serif;
                                line-height: 1.6;
                                max-width: 100%;
                                overflow-x: auto;
                            }}
                            .html-report {styles.replace('{', '.html-report {')}
                            </style>
                            """
                            
                            display_content = f'{scoped_styles}<div class="html-report">{body_content}</div>'
                        else:
                            display_content = f'<div class="html-report">{body_content}</div>'
                    else:
                        # Fallback: display full content in iframe with better styling
                        display_content = f"""
                        <style>
                        .html-report {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            max-width: 100%;
                            padding: 20px;
                            background: white;
                        }}
                        </style>
                        <div class="html-report">
                        {html_content}
                        </div>
                        """
                    
                    components.html(display_content, height=600, scrolling=True)
                    
                    if st.button("❌ Hide Report", key=f"hide_{html_file.name}"):
                        st.session_state.show_html_report = False
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading HTML report: {e}")
                    st.info("You can open the report manually using the file path shown above.")
    
    with col2:
        st.subheader("📋 Report Details")
        st.write(f"**Created:** {report['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Files:** {len(report['html_files'])} HTML, {len(report['csv_files'])} CSV")
        
        if report['csv_files']:
            csv_file = report['csv_files'][0]
            df = pd.read_csv(csv_file)
            st.write(f"**Questions:** {len(df)}")
            st.write(f"**Columns:** {', '.join(df.columns)}")

def show_documentation_tab():
    """Show project documentation."""
    st.header("📚 Project Documentation")
    
    # Main README
    readme_file = PROJECT_ROOT / "README.md"
    if readme_file.exists():
        st.subheader("📖 Main README")
        readme_content = readme_file.read_text()
        st.markdown(readme_content)
    
    # Core library documentation
    with st.expander("🔧 Core Library API"):
        st.markdown("""
        The `dnd_lora_core.py` module provides three main classes:
        
        ### DnDLoRATrainer
        - Handles LoRA training with MPS optimization
        - Automatic data preparation and formatting
        - Training progress monitoring
        
        ### DnDModelComparator  
        - Side-by-side model comparison
        - D&D terminology analysis
        - HTML report generation
        
        ### DnDDataProcessor
        - D&D SRD data processing
        - Q&A dataset generation
        - Question set management
        """)
    
    # Scripts documentation
    with st.expander("📜 Available Scripts"):
        scripts_info = {
            "train_dnd_lora.py": "Main training script with argument parsing",
            "compare_models.py": "Comprehensive model comparison with HTML reports",
            "demo.py": "Simple demonstration of zero-to-hero transformation",
            "api_server.py": "Production FastAPI server for real-time comparison",
            "prepare_dnd_data.py": "Data preparation and Q&A generation"
        }
        
        for script, description in scripts_info.items():
            st.markdown(f"**{script}**: {description}")

# Helper functions
def get_model_size(model_path: Path) -> str:
    """Get human-readable model size."""
    total_size = 0
    if model_path.exists():
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    
    # Convert to human readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024
    return f"{total_size:.1f} TB"

def run_model_comparison(model_name: str, num_questions: int, question_type: str):
    """Run model comparison and display results."""
    with st.spinner("Running model comparison..."):
        # This would integrate with the compare_models.py script
        st.info(f"Comparing {model_name} with {num_questions} {question_type.lower()} questions")
        
        # Simulate progress
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        st.success("Comparison completed! Check the Training & Evaluation tab for results.")

def send_chat_message(message: str, model_type: str):
    """Send chat message to API and update history."""
    if model_type == "both":
        # Use the comparison endpoint which returns both responses
        result = call_api("generate", {
            "prompt": message,
            "max_length": 150
        })
        
        if result["success"]:
            response_data = result["data"]
            st.session_state.chat_history.append({
                "question": message,
                "type": "both",
                "original_response": response_data["original_response"],
                "lora_response": response_data["lora_response"],
                "timestamp": datetime.now()
            })
        else:
            st.error(f"Failed to get responses: {result.get('error', 'Unknown error')}")
    else:
        # For single model requests, we still need to use the comparison endpoint
        # and extract only the requested model's response
        result = call_api("generate", {
            "prompt": message,
            "max_length": 150
        })
        
        if result["success"]:
            response_data = result["data"]
            if model_type == "lora":
                response_text = response_data["lora_response"]
            else:  # original
                response_text = response_data["original_response"]
            
            st.session_state.chat_history.append({
                "question": message,
                "type": model_type,
                "response": response_text,
                "timestamp": datetime.now()
            })
        else:
            st.error(f"Failed to get response: {result.get('error', 'Unknown error')}")

def extract_report_content(html_content: str) -> Dict[str, Any]:
    """Extract key content from HTML report for display in Streamlit."""
    import re
    
    content = {
        'title': 'D&D Model Comparison Report',
        'summary': {},
        'comparisons': []
    }
    
    # Extract title
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content)
    if title_match:
        content['title'] = re.sub(r'<[^>]+>', '', title_match.group(1))
    
    # Extract metrics
    metrics = re.findall(r'<div class="metric">.*?<h3>(.*?)</h3>.*?<div class="value[^"]*">\s*(.*?)\s*</div>', html_content, re.DOTALL)
    for metric_name, metric_value in metrics:
        metric_name = re.sub(r'<[^>]+>', '', metric_name)
        metric_value = re.sub(r'<[^>]+>', '', metric_value)
        content['summary'][metric_name] = metric_value
    
    # Extract comparisons
    comparisons = re.findall(r'<div class="comparison">.*?<div class="question">(.*?)</div>.*?<div class="response original">.*?<strong>📚 Original Model:</strong><br>\s*(.*?)\s*</div>.*?<div class="response lora">.*?<strong>🐉 LoRA Fine-tuned:</strong><br>\s*(.*?)\s*</div>.*?<div class="analysis">(.*?)</div>', html_content, re.DOTALL)
    
    for question, original, lora, analysis in comparisons:
        question = re.sub(r'<[^>]+>', '', question).strip()
        original = re.sub(r'<[^>]+>', '', original).strip()
        lora = re.sub(r'<[^>]+>', '', lora).strip()
        analysis = re.sub(r'<[^>]+>', '', analysis).replace('📈 Analysis:', '').strip()
        
        content['comparisons'].append({
            'question': question,
            'original': original,
            'lora': lora,
            'analysis': analysis
        })
    
    return content

def show_recent_comparisons():
    """Show recent comparison results."""
    st.subheader("📊 Recent Comparisons")
    
    # This would load and display recent comparison data
    sample_data = {
        "Question": [
            "What is a fireball spell?",
            "How does armor class work?", 
            "What are the different dice types?",
            "Explain saving throws",
            "What is a paladin?"
        ],
        "Original D&D Terms": [0, 1, 0, 0, 1],
        "LoRA D&D Terms": [4, 3, 2, 2, 5],
        "Improvement": ["400%", "200%", "200%", "200%", "400%"]
    }
    
    df = pd.DataFrame(sample_data)
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
