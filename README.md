# 📚 Cora Citation Network Paper Classification

A graph-based machine learning project for **research paper topic classification** using the **Cora citation network**.

The project compares a traditional machine learning baseline, **Random Forest with TF-IDF-transformed paper features**, against a **two-layer Graph Convolutional Network (GCN)** that learns from both paper features and citation relationships.

The trained GCN model is also exported to **ONNX** for deployment and inference.

---

## 🚀 Live Demo

🔗 **[Cora Research Paper Prediction — Live Application](https://cora-research-paper-prediction.onrender.com)**

The deployed application provides an interface for research-paper topic prediction using the trained model.

---

## 🎯 Project Objective

Research papers are naturally connected through citations. Traditional machine learning models can classify papers based on their individual features, but they do not naturally model the relationships between papers.

This project investigates the difference between:

* **Feature-based classification** using Random Forest
* **Graph-based classification** using Graph Convolutional Networks

The key idea behind the GCN is **message passing**: a paper can learn not only from its own feature representation but also from information propagated through connected papers in the citation network.

---

## 🧠 Problem Statement

Given a collection of research papers represented as nodes in a citation graph, the objective is to predict the **research topic/class of each paper**.

### Input

Each paper is represented using:

* A 1,433-dimensional feature vector
* Citation relationships with other papers
* A topic label for supervised learning

### Output

The model predicts one of **7 research-paper topics**.

---

# 📊 Dataset — Cora

The project uses the **Cora citation network**, loaded through the `Planetoid` dataset wrapper from PyTorch Geometric.

In the graph representation:

```text
Research Paper A ─────► Research Paper B
       ●                       ●
     Node                    Node
```

* **Node** → Research paper
* **Edge** → Citation relationship
* **Node features** → Word-based feature representation
* **Node label** → Research-paper topic

### Dataset Statistics

| Property                |  Value |
| ----------------------- | -----: |
| Research Papers / Nodes |  2,708 |
| Citation Edges          | 10,556 |
| Features per Paper      |  1,433 |
| Number of Topics        |      7 |
| Training Papers         |    140 |
| Validation Papers       |    500 |
| Testing Papers          |  1,000 |

The standard Cora train/validation/test masks provided by PyTorch Geometric are used.

---

# 🔬 Project Workflow

```text
                    Cora Dataset
                         │
                         ▼
              Citation Network Graph
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      TF-IDF Features          Node Features
             │                       │
             ▼                       ▼
    Random Forest              Two-Layer GCN
             │                       │
             ▼                       ▼
       Predictions            Message Passing
             │                       │
             └───────────┬───────────┘
                         ▼
                  Model Evaluation
                         │
                         ▼
             Accuracy + Macro-F1
                         │
                         ▼
                 GCN Model → ONNX
                         │
                         ▼
                    Deployment
```

---

# ⚙️ Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* TF-IDF transformation

### Deep Learning

* PyTorch
* PyTorch Geometric
* Graph Convolutional Network (GCN)

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn
* NetworkX

### Model Deployment

* ONNX
* ONNX Script
* Render

---

# 🧩 Model 1 — Random Forest Baseline

The first approach provides a traditional machine-learning baseline.

The original Cora binary word-indicator features are transformed using:

```python
TfidfTransformer()
```

The resulting TF-IDF representation is used to train a:

```python
RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

### Why use a baseline?

The baseline answers an important question:

> How well can research papers be classified using their individual feature representations without graph-based message passing?

The Random Forest treats each paper independently during prediction.

---

# 🕸️ Model 2 — Two-Layer Graph Convolutional Network

The second approach uses a **Graph Convolutional Network**.

The implemented architecture is:

```text
Input Features
     │
     ▼
GCNConv(1433 → 32)
     │
     ▼
ReLU
     │
     ▼
Dropout (p = 0.5)
     │
     ▼
GCNConv(32 → 7)
     │
     ▼
Class Logits
```

### Architecture

```python
class SimpleGCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.5
    ):
        ...
```

Configuration used in the notebook:

| Parameter               | Value |
| ----------------------- | ----: |
| Input Dimension         | 1,433 |
| Hidden Dimension        |    32 |
| Output Dimension        |     7 |
| Dropout                 |   0.5 |
| Optimizer               |  Adam |
| Learning Rate           |  0.01 |
| Maximum Epochs          |   200 |
| Early-Stopping Patience |    20 |

---

# 🔄 How GCN Message Passing Works

Unlike Random Forest, the GCN uses the citation graph.

Conceptually:

```text
              Paper B
                 │
                 ▼
Paper A ─────► Paper C ─────► Paper D
                 │
                 ▼
              Paper E
```

The representation of a paper can be influenced by information from its neighboring papers.

This allows the model to capture information contained in the **citation structure** in addition to the paper's own features.

The GCN performs two graph-convolution operations:

```text
Paper Features
      ↓
First Graph Convolution
      ↓
Hidden Representation
      ↓
Second Graph Convolution
      ↓
7 Topic Predictions
```

---

# 🏋️ Training Strategy

The GCN is trained using the Cora training mask.

The loss function is:

```python
F.cross_entropy(...)
```

The optimizer is:

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

### Early Stopping

Training can run for up to 200 epochs.

However, validation accuracy is monitored and training stops when there is no improvement for **20 consecutive epochs**.

In the recorded notebook run:

```text
Early stopping at epoch 35
```

The model state corresponding to the best validation accuracy is restored before final test evaluation.

---

# 📈 Results

The recorded notebook run produced the following test-set results:

| Model         |   Accuracy |   Macro-F1 |
| ------------- | ---------: | ---------: |
| Random Forest | **58.10%** | **56.98%** |
| GCN           | **79.20%** | **78.24%** |

### Recorded GCN Result

```text
GCN Test Accuracy : 0.792
GCN Test Macro-F1: 0.782
```

### Recorded Random Forest Result

```text
Random Forest Accuracy : 0.5810
Random Forest Macro-F1: 0.5698
```

These values are from the notebook's recorded run. Exact results can vary slightly depending on package versions, hardware, initialization, and training conditions.

---

# 📊 Evaluation Metrics

## Accuracy

Accuracy measures the fraction of correctly classified papers:

```text
Accuracy =
Correct Predictions / Total Predictions
```

It provides an overall measure of classification performance.

---

## Macro-F1

Macro-F1 calculates the F1 score independently for each class and then takes the average.

This gives every research-paper topic equal importance rather than allowing larger classes to dominate the metric.

---

# 🧪 Confusion Matrix

The notebook generates confusion matrices for both:

* Random Forest
* GCN

These matrices help identify which research-paper topics are frequently confused with each other.

---

# 🔍 What Did We Learn?

The experiment demonstrates the difference between feature-only machine learning and graph-based learning.

### Random Forest

The Random Forest uses the paper's transformed feature representation independently.

It does not perform graph message passing.

### GCN

The GCN incorporates the citation graph through graph convolution.

This allows information from connected papers to contribute to the learned representation.

The recorded experiment therefore provides an example where **citation-network information is useful for research-paper topic classification**.

However, this experiment alone does **not** establish that citations cause topic membership. The observed performance can also be related to graph homophily, benchmark characteristics, or properties of the standard Cora split.

---

# 📦 ONNX Model Export

After training, the GCN is exported to **Open Neural Network Exchange (ONNX)** format.

The notebook creates:

```text
simple_gcn_cora.onnx
```

The exported model uses:

```text
Input:
    node_features
    edge_indices

Output:
    logits
```

Dynamic axes are configured for:

```text
num_nodes
num_edges
```

The export uses:

```text
ONNX Opset Version: 18
```

ONNX provides a standardized representation that can be useful for deploying trained models outside the original PyTorch training environment.

---

# 🌐 Deployment

The trained prediction system is deployed using **Render**.

### Live Application

**[Open the Cora Research Paper Prediction App](https://cora-research-paper-prediction.onrender.com)**

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/AbhishekNimaje435/Cora-Research-Paper-Prediction.git
cd Cora-Research-Paper-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install torch torch-geometric scikit-learn pandas matplotlib seaborn networkx onnx onnxscript
```

---

# ▶️ Running the Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open the Cora classification notebook and execute the cells sequentially.

The Cora dataset is downloaded through the PyTorch Geometric `Planetoid` dataset interface.

---

# 📁 Main Project Components

The core research workflow contains:

```text
Cora Research Paper Classification
│
├── Cora Dataset
│
├── Feature Analysis
│
├── Citation Network Visualization
│
├── TF-IDF Transformation
│
├── Random Forest Baseline
│
├── Two-Layer GCN
│
├── Model Evaluation
│   ├── Accuracy
│   ├── Macro-F1
│   └── Confusion Matrix
│
├── ONNX Model Export
│
└── Render Deployment
```

---

# ⚠️ Important Implementation Notes

### 1. Transductive Graph Learning

The GCN receives the full Cora graph structure while training uses the provided training mask. This follows the standard transductive setup used by the Cora benchmark.

### 2. TF-IDF Baseline

The notebook applies `TfidfTransformer` to the Cora feature matrix before selecting the train/test masks.

Therefore, the README describes the implemented baseline as **TF-IDF + Random Forest** rather than claiming a separate degree/log-degree feature implementation.

### 3. Benchmark Limitations

The experiment uses the standard Cora benchmark split and a single recorded training run.

For a more rigorous research evaluation, future experiments could include:

* Multiple random seeds
* Mean ± standard deviation
* Additional graph datasets
* Different GNN architectures
* Different train/validation/test splits
* Ablation studies
* Graph-only features
* Hyperparameter tuning

---

# 🚀 Future Improvements

Possible extensions include:

* [ ] Test on CiteSeer and PubMed
* [ ] Compare GCN with GraphSAGE
* [ ] Compare GCN with GAT
* [ ] Run experiments across multiple random seeds
* [ ] Report mean ± standard deviation
* [ ] Perform hyperparameter optimization
* [ ] Add model explainability
* [ ] Add class-wise precision, recall and F1
* [ ] Investigate graph homophily
* [ ] Build a richer research-paper prediction interface
* [ ] Optimize ONNX inference
* [ ] Add automated model evaluation
* [ ] Add CI/CD for deployment

---

# 📚 Key Concepts Demonstrated

This project demonstrates practical implementation of:

```text
Machine Learning
       ↓
TF-IDF
       ↓
Random Forest
       ↓
Graph Representation
       ↓
Graph Neural Networks
       ↓
GCN Message Passing
       ↓
Model Evaluation
       ↓
ONNX Export
       ↓
Model Deployment
```

It combines **traditional machine learning, deep learning, graph representation learning, and model deployment** in one end-to-end project.

---

# 📖 References

The implementation uses the following major libraries and concepts:

* PyTorch
* PyTorch Geometric
* Scikit-learn
* NetworkX
* Pandas
* NumPy
* Matplotlib
* Seaborn
* ONNX

The Cora dataset is accessed through the **Planetoid** dataset implementation in PyTorch Geometric.

---

# 👨‍💻 Author

**Abhishek Nimaje**

B.Tech — Artificial Intelligence & Data Science

GitHub:
**https://github.com/AbhishekNimaje435**

LinkedIn:
**https://www.linkedin.com/in/abhishek-nimaje-43200636b/**

---

# ⭐ Project Summary

> **Cora Citation Network Paper Classification** demonstrates how Graph Neural Networks can leverage citation relationships for research-paper topic classification.

The project establishes a traditional **TF-IDF + Random Forest baseline**, implements a **two-layer GCN**, evaluates both approaches using **Accuracy and Macro-F1**, exports the trained GCN to **ONNX**, and deploys the prediction system through **Render**.

**Live Demo:**
https://cora-research-paper-prediction.onrender.com

