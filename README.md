# ⚙️ Contextual Predictive Maintenance using IoT Sensor Fusion and LightGBM

## 📌 Project Overview

Industrial machines continuously generate large amounts of sensor telemetry such as temperature, rotational speed, torque, and tool wear.

Traditional predictive maintenance systems often analyze these sensor readings independently. However, machine failures can also depend on the operating context in which the machine is running.

This project develops a **Contextual Predictive Maintenance System** that combines industrial IoT sensor telemetry with contextual operating information to identify potential machine failures before breakdown.

The complete pipeline includes:

- Industrial IoT Data Processing
- Sensor Signal Analysis
- Rolling Statistical Features
- Context Simulation
- Contextual Sensor Fusion
- Feature Engineering
- Class Imbalance Analysis
- SMOTE-based Resampling
- LightGBM Classification
- Model Evaluation
- Noise Robustness Testing
- Threshold Optimization
- Explainability
- Interactive Streamlit Dashboard
- GitHub-based Project Management

The final system demonstrates an end-to-end machine learning workflow for industrial predictive maintenance.

---

# 🎯 Problem Statement

Unexpected machine failures can result in:

- Production downtime
- Increased maintenance costs
- Equipment damage
- Reduced productivity
- Delayed manufacturing operations
- Safety and operational risks

Traditional maintenance strategies generally follow either:

### Reactive Maintenance

Maintenance is performed only after equipment fails.

### Preventive Maintenance

Maintenance is performed according to a predefined schedule.

Both approaches have limitations.

Predictive maintenance attempts to solve this problem by analyzing machine sensor data and identifying patterns associated with failure before the actual breakdown occurs.

This project extends the idea by incorporating both **machine telemetry and contextual operating conditions**.

---

# 💡 Business Objective

The primary objective is to develop a machine learning pipeline capable of identifying machine failure risk using industrial IoT telemetry.

The system is designed to support:

- Maintenance Engineers
- Manufacturing Companies
- Factory Operators
- Reliability Engineers
- Production Managers
- Industrial IoT Platforms

Potential business benefits include:

- Reduced unplanned downtime
- Earlier failure detection
- Better maintenance scheduling
- Reduced unnecessary maintenance
- Improved equipment availability
- Improved operational efficiency

---

# 📊 Dataset

## AI4I 2020 Predictive Maintenance Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

Official dataset source:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset contains approximately **10,000 machine observations** representing industrial operating conditions.

Important attributes include:

- UDI
- Product ID
- Product Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure
- Tool Wear Failure
- Heat Dissipation Failure
- Power Failure
- Overstrain Failure
- Random Failure

The primary target used for classification is:

```text
Machine failure
```

where:

```text
0 = Normal Operation
1 = Machine Failure
```

---

# 🧠 Why Contextual Predictive Maintenance?

Sensor readings alone may not always provide sufficient information about why a machine is approaching failure.

For example, the same sensor reading may represent different risk levels depending on:

- Factory workload
- Environmental conditions
- Machine operating state
- Temperature conditions
- Tool wear
- Production intensity

Therefore, contextual variables are combined with machine telemetry to provide additional operating information.

This project demonstrates how **sensor-context fusion** can be incorporated into a machine learning pipeline.

---

# 🏗️ Project Architecture

```text
AI4I IoT Telemetry
        │
        ▼
Data Ingestion
        │
        ▼
Data Cleaning & Validation
        │
        ▼
Sensor Signal Processing
        │
        ▼
Rolling Statistical Features
        │
        ▼
External Context Simulation
        │
        ▼
Contextual Data Fusion
        │
        ▼
Feature Engineering
        │
        ▼
Contextual Feature Analysis
        │
        ▼
Class Imbalance Analysis
        │
        ▼
Stratified Cross-Validation
        │
        ▼
SMOTE on Training Data
        │
        ▼
LightGBM Classification
        │
        ▼
Model Evaluation
        │
        ▼
Noise Robustness Testing
        │
        ▼
Threshold Optimization
        │
        ▼
Model Explainability
        │
        ▼
Streamlit Dashboard
```

---

# 🔄 Machine Learning Workflow

The overall workflow consists of four major phases.

### Phase 1 — Data and Signal Processing

Raw industrial telemetry is loaded, validated, cleaned, and transformed into useful signal-level features.

### Phase 2 — Contextual Feature Engineering

Additional contextual operating information is created and fused with machine sensor data.

### Phase 3 — Machine Learning

Class imbalance is handled carefully and a LightGBM classification model is trained and evaluated.

### Phase 4 — Robustness and Deployment

The model is tested under noisy sensor conditions, classification thresholds are analyzed, and results are presented through Streamlit.

---

# 📅 Four-Week Engineering Roadmap

## 🟢 Week 1 — IoT Telemetry and Signal Processing

### Issue #1 — Dataset Ingestion

Loaded the AI4I 2020 Predictive Maintenance dataset and validated its basic structure.

### Issue #2 — Data Cleaning

Performed data quality checks and prepared the raw telemetry for further processing.

### Issue #3 — Sensor Analysis

Analyzed important machine sensor variables and failure labels.

### Issue #4 — Signal Processing

Prepared machine telemetry for time-oriented and statistical feature extraction.

### Issue #5 — Rolling Statistical Features

Generated rolling statistics such as:

- Rolling Mean
- Rolling Standard Deviation
- Rolling Variance

These features help represent short-term changes in sensor behavior.

### Week 1 Deliverables

- Dataset ingestion pipeline
- Clean machine telemetry
- Sensor analysis
- Signal processing workflow
- Rolling statistical features

---

# 🟡 Week 2 — Context Fusion and Feature Engineering

### Issue #6 — Context Simulation

Created contextual operating variables to represent additional environmental and operational conditions.

Examples include:

- Ambient conditions
- Factory load
- Operating context

### Issue #7 — Contextual Data Fusion

Combined machine telemetry with contextual operating information.

### Issue #8 — Feature Engineering

Generated additional machine learning features from the available sensor and context variables.

### Issue #9 — Contextual Interaction Features

Created features representing interactions between machine telemetry and operating context.

### Issue #10 — Contextual Feature Analysis / Ablation Study

Analyzed the contribution of contextual information and prepared the final modeling dataset.

### Week 2 Deliverables

- Simulated contextual variables
- Sensor-context fusion pipeline
- Engineered features
- Context interaction features
- Modeling-ready dataset
- Contextual feature analysis

---

# 🟠 Week 3 — Imbalanced Classification and LightGBM

Machine failure datasets are typically highly imbalanced because normal operating observations significantly outnumber failure observations.

This makes accuracy alone an unreliable evaluation metric.

### Issue #11 — Class Imbalance Handling

Analyzed the distribution of normal and failure classes and prepared an imbalance-aware training strategy.

### Issue #12 — LightGBM Model Training

Implemented a **LightGBM classifier** for machine failure prediction.

LightGBM was selected because it:

- Performs well on structured tabular datasets
- Handles nonlinear relationships
- Supports efficient gradient boosting
- Works effectively with engineered sensor features
- Provides strong classification performance

### Issue #13 — Model Evaluation

Evaluated machine failure predictions using classification metrics.

### Issue #14 — Cross-Validation

Used stratified validation techniques to preserve the failure class distribution during evaluation.

### Issue #15 — Failure Analysis

Analyzed correct predictions, false alarms, and missed machine failures.

### Week 3 Deliverables

- Imbalance-aware training pipeline
- LightGBM classification model
- Cross-validation workflow
- Classification metrics
- Confusion matrix
- Failure prediction analysis

---

# 🔴 Week 4 — Robustness, Optimization and Deployment

## Issue #16 — Sensor Noise Simulation

Introduced controlled noise into sensor measurements to simulate imperfect industrial sensor readings.

## Issue #17 — Noise Robustness Analysis

Measured how model performance changes as sensor noise increases.

This helps determine whether the model remains stable under realistic operating conditions.

## Issue #18 — Threshold Optimization

Analyzed different classification probability thresholds.

Threshold optimization helps balance:

- Failure detection
- False alarms
- Precision
- Recall

## Issue #19 — Model Explainability

Analyzed the contribution of important features to better understand the machine failure prediction process.

## Issue #20 — Streamlit Dashboard

Developed an interactive Streamlit dashboard to demonstrate the completed predictive maintenance workflow.

### Week 4 Deliverables

- Noise simulation
- Robustness evaluation
- Threshold analysis
- Model interpretation
- Interactive Streamlit application
- Final project documentation

---

# 🤖 Machine Learning Model

## LightGBM Classifier

The primary machine learning algorithm used in this project is:

```text
LightGBM
Light Gradient Boosting Machine
```

LightGBM builds an ensemble of decision trees using gradient boosting.

It is particularly useful for this project because industrial sensor datasets contain complex nonlinear relationships between variables such as:

- Temperature
- Speed
- Torque
- Tool wear
- Operating load

---

# ⚖️ Handling Class Imbalance

Machine failure prediction is an imbalanced classification problem.

Most observations represent:

```text
Normal Operation
```

while only a small percentage represent:

```text
Machine Failure
```

Therefore, relying only on accuracy can produce misleading results.

The project uses an imbalance-aware modeling strategy and evaluates the failure class using metrics such as precision, recall, and Macro F1.

Where resampling is applied, it should be performed only on training data to avoid information leakage into validation or test data.

---

# 📈 Model Evaluation

The LightGBM model is evaluated using:

- Macro F1 Score
- Failure Precision
- Failure Recall
- Confusion Matrix

The current project dashboard reports approximately:

```text
Macro F1 Score      : 0.931
Failure Precision   : 0.879
Failure Recall      : 0.853
```

Confusion matrix:

```text
                    Predicted Normal    Predicted Failure

Actual Normal              1924                  8
Actual Failure               10                 58
```

This corresponds to:

```text
True Negatives  : 1924
False Positives : 8
False Negatives : 10
True Positives  : 58
```

---

# 🎯 Why Macro F1?

Accuracy can be misleading when the dataset is highly imbalanced.

For example, if most machines are healthy, a model predicting nearly every machine as healthy could achieve high accuracy while failing to detect actual breakdowns.

Macro F1 calculates performance across both classes more equally.

Therefore, it is an important metric for evaluating this predictive maintenance system.

---

# 🔍 Precision vs Recall

## Failure Precision

Precision answers:

> When the model predicts a machine failure, how often is that prediction correct?

Higher precision means fewer unnecessary maintenance alerts.

## Failure Recall

Recall answers:

> Of all machines that actually failed, how many failures did the model detect?

For predictive maintenance, recall is particularly important because missed failures can result in unexpected downtime and operational losses.

---

# 🧩 Confusion Matrix

The confusion matrix separates predictions into four categories.

### True Negative

The machine was healthy and correctly predicted as healthy.

### True Positive

The machine failed and the model correctly detected the failure.

### False Positive

The machine was healthy but the model generated a failure warning.

This may result in an unnecessary maintenance inspection.

### False Negative

The machine actually failed but the model predicted normal operation.

This is especially important in predictive maintenance because missed failures may result in unexpected downtime.

---

# 🔊 Noise Robustness Analysis

Real industrial IoT sensors do not always generate perfectly clean measurements.

Sensor values may be affected by:

- Measurement error
- Calibration problems
- Environmental interference
- Hardware limitations
- Communication noise

Therefore, this project evaluates model performance after introducing controlled noise into sensor measurements.

The Streamlit dashboard visualizes:

```text
Noise Level vs Macro F1 Score
```

This helps assess how robust the model is when sensor quality decreases.

---

# 🎯 Threshold Optimization

Binary classification models normally use a probability threshold to convert predicted probabilities into classes.

A default threshold may not always provide the best trade-off for predictive maintenance.

A lower threshold can:

- Detect more failures
- Increase recall
- Potentially increase false alarms

A higher threshold can:

- Reduce false alarms
- Potentially miss real failures

Therefore, threshold analysis helps identify a more appropriate balance between operational risk and maintenance cost.

---

# 💡 Contextual Sensor Fusion

One of the main concepts explored in this project is combining machine telemetry with contextual information.

Instead of relying only on:

```text
Sensor Data
```

the project explores:

```text
Sensor Data
+
Operating Context
=
Context-Aware Failure Features
```

This allows the predictive pipeline to consider both the physical machine measurements and the conditions under which those measurements occurred.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application.

The dashboard contains the following pages:

## 🏠 Overview

Displays:

- Project objective
- Project duration
- GitHub issue count
- Machine learning model
- Primary KPI
- Complete project pipeline

## 🚨 Failure Risk Demo

Allows users to enter:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Factory Load

The hosted demonstration calculates a transparent risk score and categorizes the result into:

```text
Low Risk
Medium Risk
High Risk
```

The demonstration formula is separate from the trained model and is intended to illustrate the end-user workflow when model artifacts are not deployed with the repository.

## 📊 Model Metrics

Displays:

- Macro F1 Score
- Failure Precision
- Failure Recall
- Confusion Matrix
- True Positives
- True Negatives
- False Positives
- False Negatives
- Business interpretation

## 🔊 Noise Robustness

Displays:

- Noise experiment results
- Noise level
- Macro F1 performance
- Performance trend visualization

## 🎯 Threshold Analysis

Displays classification threshold experiment results and explains the trade-off between failure detection and false alarms.

## 🗂 Project Roadmap

Displays the complete four-week engineering workflow covering all 20 GitHub Issues.

---

# 🛠️ Technology Stack

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- LightGBM

## Imbalanced Learning

- Imbalanced-learn
- SMOTE

## Model Persistence / Utilities

- Joblib

## Visualization & Reporting

- Streamlit
- Pandas visualization utilities

## Development & Version Control

- Git
- GitHub
- VS Code

---

# 📁 Repository Structure

```text
contextual-predictive-maintenance/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── ai4i2020.csv
│   └── processed/
│
├── models/
│
├── reports/
│   ├── lightgbm_metrics.json
│   ├── noise_robustness.csv
│   └── threshold_metrics.json
│
├── src/
│   ├── data_ingestion.py
│   ├── signal_processing.py
│   ├── context_fusion.py
│   ├── feature_engineering.py
│   ├── ablation_study.py
│   ├── train_lightgbm.py
│   ├── evaluate_model.py
│   ├── noise_analysis.py
│   ├── threshold_tuning.py
│   └── explainability.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> The exact generated files available in `reports/` depend on which experiment scripts have been executed.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/contextual-predictive-maintenance.git
```

Move into the project:

```bash
cd contextual-predictive-maintenance
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Typical project dependencies include:

```text
pandas
numpy
scikit-learn
imbalanced-learn
lightgbm
joblib
streamlit
```

Keep `requirements.txt` synchronized with the imports actually used by the project.

---

# ▶️ Running the Machine Learning Pipeline

Individual scripts can be executed from the project root.

Example:

```bash
python src/data_ingestion.py
```

Then run the required preprocessing and feature engineering scripts.

Train LightGBM:

```bash
python src/train_lightgbm.py
```

Run evaluation and experiment scripts as required:

```bash
python src/evaluate_model.py
python src/noise_analysis.py
python src/threshold_tuning.py
```

---

# 🌐 Running the Streamlit Application

From the repository root:

```bash
streamlit run app/streamlit_app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# 🔐 Git Ignore Strategy

Large datasets, trained model files, virtual environments, and temporary development files should not be committed unnecessarily.

Example `.gitignore` rules:

```gitignore
# Virtual Environment
venv/
.venv/

# Python
__pycache__/
*.pyc

# IDE
.vscode/

# Jupyter
.ipynb_checkpoints/

# Dataset
data/

# Models
models/
*.pkl
*.joblib

# Environment
.env
```

Report files required by the deployed Streamlit application should remain available if the app depends on them at runtime.

---

# 🔄 GitHub Development Workflow

This project follows a structured version-control workflow.

Development activities include:

- GitHub Issues
- GitHub Projects Kanban Board
- Week-wise development
- Semantic commit messages
- Issue-linked commits
- Regular repository updates
- README documentation

Example commit messages:

```text
feat: add rolling sensor statistics (fixes #5)

feat: implement contextual data fusion (fixes #7)

feat: train LightGBM failure classifier (fixes #12)

test: evaluate sensor noise robustness (fixes #17)

feat: optimize failure classification threshold (fixes #18)

feat: build Streamlit predictive maintenance dashboard (fixes #20)
```

---

# 🗂️ GitHub Project Management

The development roadmap is divided into four weeks.

```text
Week 1
Issues #1–#5
IoT Telemetry & Signal Processing

Week 2
Issues #6–#10
Context Fusion & Feature Engineering

Week 3
Issues #11–#15
Imbalanced Classification & LightGBM

Week 4
Issues #16–#20
Noise Robustness, Optimization & Deployment
```

Tasks can be tracked using:

```text
To Do → In Progress → Done
```

This provides a clear history of project development.

---

# 🏭 Potential Real-World Applications

This type of predictive maintenance system can be extended to:

- Manufacturing equipment
- CNC machines
- Industrial motors
- Pumps
- Compressors
- Production lines
- Factory automation systems
- Industrial IoT platforms

---

# 🔮 Future Improvements

The project can be extended with:

### Real-Time IoT Streaming

Integrate MQTT, Kafka, or another streaming platform for live machine telemetry.

### Time-Series Models

Explore:

- LSTM
- GRU
- Temporal CNN
- Transformer-based models

### Explainable AI

Extend model interpretation using techniques such as SHAP.

### Real Model Deployment

Deploy the serialized LightGBM model behind the Streamlit application or an API so the risk demo performs live model inference.

### Cloud Deployment

Deploy the application using cloud infrastructure.

### Maintenance Alerts

Integrate:

- Email notifications
- SMS alerts
- Maintenance tickets

### Cost-Sensitive Learning

Assign higher penalties to missed failures compared with false alarms.

### Model Monitoring

Monitor prediction drift and sensor distribution changes after deployment.

---

# 📌 Key Learning Outcomes

Through this project, I gained practical experience in:

- Industrial IoT data processing
- Data preprocessing
- Sensor feature engineering
- Rolling statistical analysis
- Context simulation
- Contextual data fusion
- Imbalanced classification
- SMOTE
- LightGBM
- Stratified validation
- Classification metrics
- Confusion matrix analysis
- Noise robustness testing
- Threshold optimization
- Streamlit dashboard development
- Git and GitHub
- Issue-based project management
- End-to-end machine learning workflows

---

# 🎓 Project Summary

This project demonstrates an end-to-end **Contextual Predictive Maintenance** workflow that combines industrial IoT telemetry with contextual operating information.

The pipeline progresses from:

```text
Raw IoT Data
     ↓
Signal Processing
     ↓
Context Fusion
     ↓
Feature Engineering
     ↓
Imbalance Handling
     ↓
LightGBM
     ↓
Evaluation
     ↓
Robustness Testing
     ↓
Threshold Optimization
     ↓
Streamlit Deployment
```

The final system provides both a machine learning experimentation workflow and an interactive dashboard for communicating predictive-maintenance results.

---

# 👨‍💻 Author

**Vatsal Dhuvad**

Data Science & Machine Learning Intern  
**Infotact Solutions**

Computer Engineering Student  
Gandhinagar, Gujarat, India

Career Interests:

- Data Science
- Machine Learning
- Artificial Intelligence
- Predictive Analytics
- Industrial AI

---

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for additional information.

---

# ⭐ Acknowledgement

Special thanks to **Infotact Solutions** for providing the project framework and internship opportunity to gain practical experience in Data Science and Machine Learning.

Dataset:

**AI4I 2020 Predictive Maintenance Dataset**  
UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
