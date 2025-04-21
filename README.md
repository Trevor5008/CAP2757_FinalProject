# CAP2757 - Final Project - NY Housing Market Analysis

## Overview

A Streamlit web app that provides data analysis, hypothesis testing and machine learning to NY Housing Market Data

## Team

- Carlos
- Luis
- Trevor

## Objectives

- Frame an EDA-driven analysis of the NY Housing Market
- Clean and structure raw data for ML model training and output
- Build a Streamlit UI for users to interact with analyses and predictions
- Demonstrate the predictive capabilities of ML modeling

# Instructions to Run the Streamlit App

## ✅ Prerequisites

Before running the app, it's recommended to create a virtual environment and install all required dependencies using the provided `requirements.txt` file.

### 1. Create a virtual environment (optional but recommended)

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

Ensure `requirements.txt` is in the same directory as `main.py` and `NY-House-Dataset.csv`, then run:

```bash
pip install -r requirements.txt
```

Also, make sure the file `NY-House-Dataset.csv` is placed in the **same directory** as `main.py`.

## ▶️ Running the App

1. Open your terminal (Command Prompt, Terminal, or Anaconda Prompt).
2. Navigate to the folder containing `main.py`:
   ```bash
   cd path/to/your/folder
   ```
3. Run the Streamlit app using the following command:
   ```bash
   streamlit run main.py
   ```

After a few seconds, your browser will open the app at `http://localhost:8501`.

---

If you encounter any issues, make sure all dependencies are installed and the CSV file is correctly placed.
