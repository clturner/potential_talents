# Potential Talents

This project is designed to analyze and rank potential talent candidates for human resources roles based on provided data. It includes data exploration, feature engineering, and candidate ranking functionalities.

## Project Structure
```plaintext
potential_talents
├── data
│   ├── potential-talents - Aspiring human resources - seeking human resources.csv            
│   ├── potential-talents - Aspiring human resources - seeking human resources - appended.csv
├── notebooks
│   ├── data_exploration.ipynb        # Notebook for exploring the dataset
│   └── rank_candidates.ipynb         # Notebook for ranking candidates
├── src
│   ├── init.py                       # Initializes the source package
│   ├── utils.py                      # Utility functions
│   ├── feature_engineering_utils.py  # Feature engineering utilities
│   └── prediction_evaluation.py      # Prediction and evaluation logic
├── requirements.txt                  # Project dependencies
|── README.md
```

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- (Optional) Virtual environment tool like `venv` or `virtualenv`

## Setup Instructions

Follow these steps to clone the repository, install dependencies, and set up the environment.

### 1. Clone the Repository

Clone the project to your local machine using the following command:

```bash
git clone https://github.com/your-username/potential_talents.git
cd potential_talents
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment to keep dependencies isolated:

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
Once activated, your terminal prompt should change to indicate the virtual environment is active (e.g., (venv)).

### 3. Install Requirements
Install the required Python packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```
### 4. Verify Setup
To ensure everything is set up correctly, you can open the Jupyter notebooks in the notebooks directory:
```bash
jupyter notebook
```
Then, navigate to data_exploration.ipynb or rank_candidates.ipynb in your browser and run the cells.
### 5. Deactivate the Virtual Environment (When Done)
When you're finished working on the project, deactivate the virtual environment:
```bash
deactivate
```
## Usage
1. **Data Exploration:** Use notebooks/data_exploration.ipynb to explore the dataset and understand its structure.
2. **Candidate Ranking:** Use notebooks/rank_candidates.ipynb to rank potential candidates based on the provided logic.
3. **Source Code:** The src/ directory contains reusable Python modules for utilities, feature engineering, and evaluation.