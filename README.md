[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
# 🧠🔗 CoPCA: Capturing Symbolic Knowledge of Constraints and Incompleteness to Guide Inductive Learning in Neuro-Symbolic Knowledge Graph Completion
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
├── requirements.txt       # Necessary dependencies  
└── README.md             
```

---
## 📊 Benchmark Statistics

| **KG Size** | **Benchmark**     | **#Triples** | **#Entities** | **#Relations** |
|-------------|-------------------|--------------|----------------|----------------|
| **Large**   | DB100K            | 695,572      | 99,604         | 470            |
| **Medium**  | YAGO3-10          | 1,080,264    | 123,086        | 37             |
| **Small**   | French Royalty    | 10,526       | 2,601          | 12             |

| **KG Size** | **Benchmark**     | **#Constraints** | **#Valid** | **#Invalid** |
|-------------|-------------------|------------------|------------|--------------|
| **Large**   | DB100K            | 6                | 390,351    | 62,024       |
| **Medium**  | YAGO3-10          | 4                | 393,205    | 58,719       |
| **Small**   | French Royalty    | 2                | 1,922      | 298          |

## 📈 Evaluation Metrics

We evaluate KG completion using embedding models:
- **TransE**, **TransH**, **TransD**
- **RotatE**, **ComplEx**, **TuckER**
- **CompGCN**

Metrics reported:
- **Hits@1**, **Hits@3**, **Hits@5**, **Hits@10**
- **Mean Reciprocal Rank (MRR)**

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/SDM-TIB/CoPCA.git
```

### 2. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Benchmarks Included

- **French Royalty** KG
- **YAGO3-10**
- **DB100K**

Each benchmark includes original, validated, and constraint-filtered variants of the KG. Find DB100K and YAGO3-10 benchmarks in Leibniz Data Manager: 
https://doi.org/10.57702/y3f76e2h

---

## 🔍 Referenced Works

- **PyKEEN** – _Ali et al., 2021: [paper](http://jmlr.org/papers/v22/20-825.html)   
- **SPaRKLE** – Purohit et al., 2023: [paper](https://doi.org/10.1145/3587259.3627547)   
- **VISE** – Purohit et al., 2024:  [paper](https://ceur-ws.org/Vol-3831/)
- **VANILLA** – Purohit et al., 2025: [paper](https://doi.org/10.1016/j.knosys.2025.113939)  

---

## 👨‍💻 Authors & Contact

CoPCA has been developed by members of the Scientific Data Management Group at TIB, as an ongoing research effort.
The development is co-ordinated and supervised by Maria-Esther Vidal.
Developed and maintained by:

- **Disha Purohit and Yashrajsinh Chudasama**
- Feel free to reach out for any issues related to reproducibility or implementation at: **disha.purohit@tib.eu**

---


## ✅ License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

This project builds upon contributions and tools from the neuro-symbolic and knowledge representation communities.
