# AmbedkarGPT-Intern-Task

This repository contains a Command-Line Retrieval-Augmented Generation (RAG) QA system and an evaluation framework for the KalpIT AI Intern assignments. The system ingests a curated document corpus of Dr. B. R. Ambedkar’s works, retrieves relevant context using vector search, and answers user questions using a local large language model. Evaluation is performed on a set of 25 QA pairs, exploring chunking strategies and using standard metrics.

## Directory Structure

AmbedkarGPT-Intern-Task/
├── main.py # Assignment 1: Command-line QA system
├── evaluation.py # Assignment 2: Evaluation + metrics
├── speech.txt # Source text for Assignment 1
├── corpus/ # 6 text files for Assignment 2
├── testdataset.json # List of QA pairs for evaluation
├── testresults.json # Output after evaluation
├── README.md # Setup, usage, and notes
├── resultsanalysis.md # Written analysis of results and findings
├── requirements.txt
├── .gitignore



## Setup Instructions

**Clone the repository:**
git clone https://github.com/your-username/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task



**Create and activate a Python virtual environment (Python 3.8+):**
python3 -m venv .venv
source .venv/bin/activate



**Install dependencies:**
pip install -r requirements.txt


**Install Ollama and pull the Mistral model:**
- Download Ollama for Mac or Linux: [https://ollama.com/download](https://ollama.com/download)
- Start Ollama, then run:
    ```
    ollama pull mistral
    ```
- Make sure `ollama serve` is running before starting the QA system.

## Running the QA System (Assignment 1)
python main.py

- Type questions related to the content in `speech.txt`, e.g.:
  > What is the real remedy for caste system?
- Type `exit` to quit.

## Running Evaluation (Assignment 2)

> **Note:** For resource-constrained systems, the evaluation is run on a subset of questions and/or chunking configs.

1. Ensure the `corpus/` folder and `testdataset.json` are present in the root directory.
2. Open `evaluation.py` and optionally restrict the number of questions or chunking strategy to avoid crashes (see code comments).
3. Run:
    ```
    python evaluation.py
    ```
4. Results will be written to `testresults.json`.

## Evaluation Notes & Hardware Constraints
- By default, only the `"small"` chunk config and the first 5 questions are processed per run, to avoid system crashes. You can process more if system resources allow.
- The code supports full evaluation on all 25 questions and multiple chunk sizes—simply uncomment more configs or increase question count if you have sufficient memory/CPU.
- See `resultsanalysis.md` for detailed metric results and findings.

## Troubleshooting
- **Import errors:** Ensure your `.venv` is activated and all required Python packages are installed (`pip install -r requirements.txt`).
- **Evaluation crashes/freezes:** Run one chunk config at a time, or limit the number of questions processed per run in `evaluation.py`. Document these steps in your submission.
- **Ollama errors:** Make sure the Ollama server is running and the required model (`mistral`) is available.

## Requirements
- Python 3.8+
- All packages listed in `requirements.txt`
- Ollama, with Mistral model pulled locally (`ollama pull mistral`)
