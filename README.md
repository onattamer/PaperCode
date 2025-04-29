# CollabPersona: Anonymized Codebase

This repository contains the anonymized code for the CollabPersona framework, designed to study collaborative reasoning among persona-driven Large Language Model (LLM) agents.

The project evaluates agent behavior across three setups:
- **Scenario 1:** Solo reasoning (no feedback)
- **Scenario 2:** Collaborative reasoning (one-shot feedback)
- **Hybrid:** Collaborative vs. Solo matchups

---

## 🔹 Folder Structure

```text
PaperCode/
├── Scenario1/       # Solo reasoning experiments
├── Scenario2/       # Collaborative reasoning experiments
├── hybrid/          # Hybrid experiments
Each folder contains its own code and experiment scripts.

🔹 How to Run
Navigate into the corresponding folder (Scenario1/, Scenario2/, or hybrid/).

Run experiments: bash run_experiment.sh
Run evaluations: bash run_all_evaluations.sh
Evaluation outputs (plots and CSV files) will be saved inside eval_* directories.

🔹 Notes
All code has been anonymized for double-blind review.

No API keys, author names, or affiliations are included.