# D&D SRD LoRA Fine-Tuning Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LoRA](https://img.shields.io/badge/LoRA-PEFT-green.svg)](https://github.com/huggingface/peft)

A complete demonstration of **"zero-to-hero" knowledge injection** using LoRA (Low-Rank Adaptation) fine-tuning to transform general language models into D&D 5e experts.

## 🎯 Project Overview

This project showcases how LoRA fine-tuning can inject domain-specific knowledge into small language models with minimal computational cost. We transform models like DistilGPT2 (82M parameters) from having zero D&D knowledge to expert-level understanding using only **1.42% of the model's parameters**.

### 🏆 Key Achievements

- **Dramatic Knowledge Injection**: 200-800% increase in D&D terminology usage
- **Efficient Training**: Only ~1% of model parameters needed for domain expertise
- **Complete Pipeline**: End-to-end system from data preparation to deployment
- **Real-time Comparison**: Interactive API and dashboard for model evaluation
- **Comprehensive Evaluation**: Automated testing and HTML report generation

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd dnd_srd_model

# Install dependencies
pip install -r requirements.txt

# D&D SRD data is already included in ./data/raw/
```

### 2. Prepare Training Data

```bash
# Convert D&D SRD to training format from local data
python prepare_dnd_data.py

# This creates data/dnd_srd_qa.jsonl with 2,953 Q&A examples
```

### 3. Train a LoRA Model

```bash
# Train DistilGPT2 with D&D knowledge (fastest)
python train_dnd_lora.py --model distilgpt2 --epochs 3

# Train Pythia-1.4B (more powerful)
python train_dnd_lora.py --model EleutherAI/pythia-1.4b --epochs 2
```

### 4. Run Demonstration

```bash
# See the dramatic transformation
python demo.py

# Comprehensive comparison
python compare_models.py

# Start interactive API
python api_server.py
```

## 📁 Project Structure

```
dnd_srd_model/
├── 🧠 Core Library
│   └── dnd_lora_core.py          # Consolidated functionality
├── 🛠️ Main Scripts
│   ├── prepare_dnd_data.py       # Data preparation
│   ├── train_dnd_lora.py         # LoRA training
│   ├── compare_models.py         # Model evaluation
│   ├── demo.py                   # Simple demonstration
│   └── api_server.py             # FastAPI server
├── 🖥️ Web Interface
│   ├── streamlit_dashboard.py    # Comprehensive dashboard
│   └── launch_dashboard.py       # Dashboard launcher
├── 📊 Data & Models
│   ├── data/                     # Training datasets
│   ├── models/                   # Trained LoRA adapters
│   └── eval/                     # Evaluation reports
├── 🗂️ Archive
│   └── scripts_archive/          # Legacy development scripts (organized)
│       ├── training/             # Original training experiments
│       ├── comparison/           # Model comparison approaches
│       ├── demos/                # Various demonstration scripts
│       ├── api/                  # API server iterations
│       ├── testing/              # Quick tests and experiments
│       └── utilities/            # Data prep and utility scripts
└── 📚 Documentation
    ├── README.md                 # This file
    ├── FINAL_PROJECT_REPORT.md   # Detailed results
    └── PROJECT_COMPLETE.md       # Full documentation
```

## 🎮 Usage Examples

### Training Different Models

```bash
# Quick training on DistilGPT2 (recommended for testing)
python train_dnd_lora.py --model distilgpt2 --epochs 3 --batch-size 8

# Full training on Pythia (better results)
python train_dnd_lora.py --model EleutherAI/pythia-1.4b --epochs 2 --batch-size 4

# Custom configuration
python train_dnd_lora.py \
    --model distilgpt2 \
    --data data/dnd_srd_qa.jsonl \
    --output models/my-dnd-model \
    --epochs 5 \
    --learning-rate 3e-4 \
    --lora-rank 32
```

### Model Comparison

```bash
# Basic comparison (5 questions)
python compare_models.py --model distilgpt2

# Comprehensive evaluation
python compare_models.py \
    --model distilgpt2 \
    --batch-size 20 \
    --include-general \
    --output eval/detailed_comparison

# Quick demo
python demo.py --questions 3 --pause 2
```

### API Server

```bash
# Start server (default: localhost:8000)
python api_server.py

# Custom configuration
python api_server.py \
    --model distilgpt2 \
    --lora-path models/my-dnd-model \
    --port 8080 \
    --host 0.0.0.0
```

### Interactive Dashboard

```bash
# Launch comprehensive Streamlit dashboard
python launch_dashboard.py

# Custom host/port
python launch_dashboard.py --host 0.0.0.0 --port 8080

# Direct streamlit command
streamlit run streamlit_dashboard.py
```

The dashboard provides:
- 🏠 **Overview**: Project stats and model information
- 🤖 **Model Comparison**: Side-by-side evaluation interface
- 💬 **Interactive Chat**: Real-time model testing
- 📈 **Training & Evaluation**: Progress visualization and reports
- 📚 **Documentation**: Complete project documentation

## 🔬 Core Library API

The `dnd_lora_core.py` module provides three main classes:

### DnDLoRATrainer
```python
from dnd_lora_core import DnDLoRATrainer

trainer = DnDLoRATrainer(model_name="distilgpt2")
trainer.setup_lora(rank=16, alpha=32)
dataset = trainer.prepare_dataset("data/dnd_srd_qa.jsonl")
trainer.train(dataset, num_epochs=3)
```

### DnDModelComparator
```python
from dnd_lora_core import DnDModelComparator

comparator = DnDModelComparator(
    model_name="distilgpt2",
    lora_path="models/dnd-lora"
)
result = comparator.compare_responses("What is a Fireball spell?")
```

### DnDDataProcessor
```python
from dnd_lora_core import DnDDataProcessor

# Load from default local data/raw directory
srd_data = DnDDataProcessor.load_srd_data()

# Or specify custom path
srd_data = DnDDataProcessor.load_srd_data("path/to/raw/data")

qa_pairs = DnDDataProcessor.create_qa_pairs(srd_data)
DnDDataProcessor.save_dataset(qa_pairs, "data/training.jsonl")
```

## 📊 Training Results

### DistilGPT2 Results
- **Parameters Trained**: 1.18M / 83M (1.42%)
- **Training Loss**: 2.04 → 1.64
- **D&D Term Usage**: 200-800% increase
- **Training Time**: ~15 minutes (Apple M4)

### Pythia-1.4B Results
- **Parameters Trained**: 6.29M / 1.4B (0.44%)
- **Training Loss**: 1.92 → 1.47
- **D&D Term Usage**: 300-1000% increase
- **Training Time**: ~45 minutes (Apple M4)

## 🎯 Example Transformations

### Question: "What is a Fireball spell in D&D?"

**Original DistilGPT2:**
> "I don't know what a fireball spell is, but I think it's something that can be used to create fire."

**LoRA Fine-tuned:**
> "Fireball is a 3rd-level evocation spell. It deals 8d6 fire damage in a 20-foot radius sphere. Creatures in the area make a Dexterity saving throw for half damage."

**Analysis**: 0 → 7 D&D terms ✨ **Zero-to-Hero transformation!**

## 🛡️ Technical Details

### LoRA Configuration
- **Rank (r)**: 8-32 (controls adapter size)
- **Alpha**: 16-64 (scaling factor)
- **Target Modules**: Attention and feed-forward layers
- **Dropout**: 0.1
- **Task Type**: Causal Language Modeling

### Training Setup
- **Device**: Apple Silicon MPS optimization
- **Precision**: Float32 (MPS compatibility)
- **Data Format**: Instruction-following Q&A pairs
- **Evaluation**: 10% holdout set

### Hardware Requirements
- **Minimum**: 8GB RAM, Apple Silicon or CUDA GPU
- **Recommended**: 16GB RAM for Pythia training
- **Storage**: ~2GB for models and data

## 🌐 API Endpoints

### POST /generate
Compare responses from both models:
```json
{
  "prompt": "What is a Beholder in D&D?",
  "max_length": 150,
  "temperature": 0.7
}
```

### GET /health
Check server and model status.

### GET /docs
Interactive API documentation (Swagger UI).

## 📈 Evaluation Metrics

- **D&D Term Count**: Domain-specific vocabulary usage
- **Response Length**: Detailed vs. generic responses
- **Knowledge Accuracy**: Correctness of D&D facts
- **General Knowledge**: Preservation of non-domain abilities

## 🧹 Development Notes

This is the **refactored, clean version** of the project. The original development created many experimental scripts in the `scripts/` directory. The core functionality has been consolidated into:

1. **Core Library**: `dnd_lora_core.py`
2. **Main Scripts**: Clean, documented, production-ready
3. **Legacy Scripts**: Original development scripts (for reference)

## 🚀 Future Enhancements

- [ ] Multi-domain knowledge injection
- [ ] Larger model support (7B+ parameters)
- [ ] Advanced evaluation metrics
- [ ] Web interface for model comparison
- [ ] Docker deployment

## 📚 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [D&D 5e System Reference Document](https://dnd.wizards.com/resources/systems-reference-document)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Transformers Library](https://huggingface.co/docs/transformers/)

## 📄 License

This project is for educational and research purposes. D&D content is used under the Open Gaming License.

---

**🎲 Ready to inject domain knowledge into your language models? Start with the Quick Start guide above!**
