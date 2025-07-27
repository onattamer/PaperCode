# CollabPersona: A Framework for Collaborative Decision Analysis in Persona-Driven LLM-Based Multi-Agent Systems

## 🙏 Acknowledgment & Code Attribution

This repository builds upon the open-source [ALYMPICS framework](https://github.com/microsoft/Alympics/tree/main) introduced in:

> **Shaoguang Mao, Yuzhe Cai, Yan Xia, Wenshan Wu, Xun Wang, Fengyi Wang, Qiang Guan, Tao Ge, and Furu Wei.**  
> *ALYMPICS: LLM Agents Meet Game Theory.*  
> arXiv preprint arXiv:2311.03220, 2023. [https://arxiv.org/abs/2311.03220](https://arxiv.org/abs/2311.03220)

We thank the authors for sharing their platform.  
While our framework reuses some structural elements (e.g., `SandboxPlayground`, `AgentPlayer`), we have extensively modified and expanded the system to support:
- Persona-driven strategic behavior modeling
- One-shot feedback communication
- Scenario-based evaluation of collaborative vs. solo reasoning
- Enhanced evaluation metrics and visualizations


## 📄 Citation

**Authors:** Tamer Onat¹, Abdurrahman Gümüş¹  
**Affiliation:** ¹Izmir Institute of Technology (IZTECH)  
**Conference:** IEEE MLSP 2025 (Poster Presentation)

```bibtex
@inproceedings{onat2025collabpersona,
  title={CollabPersona: A Framework for Collaborative Decision Analysis in Persona-Driven LLM-Based Multi-Agent Systems},
  author={Onat, Tamer and G{\"u}m{\"u}{\c{s}}, Abdurrahman},
  booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year={2025}
}
```

## 🎯 Overview

This framework evaluates the effectiveness of collaborative decision-making in persona-driven multi-agent systems powered by Large Language Models (LLMs). We analyze agent behavior across three distinct experimental setups to understand how collaboration affects reasoning quality and decision outcomes.

## 🧪 Experimental Scenarios

| Scenario | Description | Feedback Type |
|----------|-------------|---------------|
| **Scenario 1** | Solo reasoning | No feedback |
| **Scenario 2** | Collaborative reasoning | One-shot feedback |
| **Hybrid** | Collaborative vs. Solo | Comparative analysis |

## 📁 Project Structure

```
PaperCode/
├── README.md                    # This file
├── Scenario1/                   # Solo reasoning experiments
│   ├── run_experiment.sh        # Main experiment runner
│   ├── run_all_evaluations.sh   # Evaluation pipeline
│   └── eval_*/                  # Generated evaluation outputs
├── Scenario2/                   # Collaborative reasoning experiments
│   ├── run_experiment.sh        
│   ├── run_all_evaluations.sh   
│   └── eval_*/                  
├── hybrid/                      # Hybrid scenario experiments
│   ├── run_experiment.sh        
│   ├── run_all_evaluations.sh   
│   └── eval_*/                  
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages: `pip install -r requirements.txt`

### Running Experiments

1. **Choose your scenario:**
   ```bash
   cd Scenario1/    # for solo reasoning
   cd Scenario2/    # for collaborative reasoning  
   cd hybrid/       # for hybrid experiments
   ```

2. **Run the experiment:**
   ```bash
   bash run_experiment.sh
   ```

3. **Generate evaluations:**
   ```bash
   bash run_all_evaluations.sh
   ```

4. **View results:**
   - Plots and CSV files will be saved in the `eval_*` directories
   - Each scenario folder contains its own evaluation outputs

## 📊 Output

Each experiment generates:
- **Quantitative metrics** (CSV format)
- **Visualization plots** (PNG/PDF format)  
- **Detailed logs** for analysis


## ✉️ Contact

For questions or collaborations:
- **Tamer Onat**: [onattamer55@gmail.com](mailto:onattamer55@gmail.com)
- **Abdurrahman Gümüş**: [abdurrahmangumus@iyte.edu.tr](mailto:abdurrahmangumus@iyte.edu.tr)
