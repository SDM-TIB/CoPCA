# 🧠🔗 COPCA: Capturing Symbolic Knowledge of Constraints and Incompleteness to Guide Inductive Learning in Neuro-Symbolic Knowledge Graph Completion
Welcome to the official repository for **CoPCA**, a novel framework that integrates symbolic constraints and incomplete knowledge to guide neuro-symbolic learning. This pipeline enhances the quality of Knowledge Graph Embeddings (KGEs) through logical rule mining, heuristic categorization, and constraint-based learning — paving the way for more explainable and robust downstream tasks such as link prediction.

---
## 📚 Overview

The **CoPCA Pipeline** follows these major steps:

1. **Validation** of the Knowledge Graph (KG) using SHACL constraints.
2. **Mining** of Horn rules over the KG using AMIE.
3. **CoPCA model**: logical rules into valid and invalid heuristics.
4. **Transformation** of the input KG into a refined KG′ using symbolic knowledge.
5. **Numerical Knowledge Graph Embedding** of KG′ using state-of-the-art KGE models.
6. **Downstream Tasks: Link Prediction** such as link prediction on vectorized KG representations.

> Example downstream task: Predicting whether a football player (yago:Ronaldo) is affiliated with a particular sports team (yago:Portugal).

---
## 📁 Repository Structure


```
├── KG/                     # Original, valid, and invalid KGs for benchmarks
│   ├── french_royalty/
│   ├── YAGO3-10/
│   └── DB100K/
│
├── Rules/                 # AMIE-mined Horn rules over KGs
│
├── Constraints/           # SHACL constraints for respective KGs
│
├── Symbolic Learning/     # Scripts for heuristic transformation and categorization
│
├── Numerical Learning/    # KGE pipeline scripts (kge.py) and configs (input.json)
├── requirements.txt
└── README.md              # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone
```

### 2. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the CoPCA Pipeline

#### 🔍 Step 1: Validate KG with SHACL
```bash
cd Constraints
python validate_kg.py --kg ../KG/YAGO3-10/original.ttl --shacl yago_constraints.ttl
```

#### 📜 Step 2: Mine Logical Rules with AMIE
```bash
cd Rules
java -jar amie.jar ../KG/YAGO3-10/original.tsv > amie_rules.txt
```

#### 🧩 Step 3: Categorize and Transform Heuristics
```bash
cd Symbolic\ Learning
python categorize_rules.py --rules ../Rules/amie_rules.txt
python transform_kg.py --kg ../KG/YAGO3-10/original.tsv --heuristics valid_rules.txt
```

#### 🔢 Step 4: Learn KGE Models
```bash
cd ../Numerical\ Learning
python kge.py --config input.json
```

---

## 📊 Benchmarks Included

- **French Royalty** KG
- **YAGO3-10**
- **DB100K**

Each benchmark includes original, validated, and constraint-filtered variants of the KG.

---

## 🔍 Referenced Works

- **PyKEEN** – _Ali et al., 2021: [http://jmlr.org/papers/v22/20-825.html](paper)   
- **SPaRKLE** – Purohit et al., 2023: [https://doi.org/10.1145/3587259.3627547](paper)   
- **VISE** – Purohit et al., 2024:  [https://ceur-ws.org/Vol-3831/](paper)
- **VANILLA** – Purohit et al., 2025: [https://doi.org/10.1016/j.knosys.2025.113939](paper)  

---

## 👨‍💻 Authors & Contact

Developed and maintained by:

- **Disha Purohit and Yashrajsinh Chudasama**
- Feel free to reach out for any issues related to reproducibility or implementation at: **disha.purohit@tib.eu**

---


## ✅ License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

This project builds upon contributions and tools from the neuro-symbolic and knowledge representation communities.
