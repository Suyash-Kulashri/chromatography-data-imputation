## Chromatography Hyperparameter Tuning

This project explores chromatography datasets, focusing on cleaning, imputing, and tuning model hyperparameters for downstream analysis. It combines Python scripts for data preparation with Jupyter notebooks that document the experimentation workflow.

### Project Structure
- `raw_data/`: Original source files, including `chromatography_final_merged_data.csv`.
- `synthetic_data/`: Scripts and outputs for generating imputed or synthetic datasets.
  - `synthetic_distribution_imputer.py`: Implements distribution-preserving imputation utilities.
  - `data_cleaning.py`: Example script that imputes selected columns and saves the cleaned output.
- `RUN_HYPERPARAMETER_TUNING_NOTEBOOK.md`: Step-by-step guide for executing the main notebook.
- `hyperparameter_tuning.ipynb`: Primary notebook for feature exploration and model tuning.
- `hyperparameter_tuning-executed.ipynb`: Notebook with captured outputs (generated after running).
- `requirements.txt`: Minimal dependencies required across scripts and notebooks.

### Environment Setup
1. Ensure Python 3.12 (or later) is available.
2. Install base dependencies:
   ```
   python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```
3. For notebook execution, additional packages such as `nbconvert`, `jupyter`, and visualization libraries may be required. Refer to `RUN_HYPERPARAMETER_TUNING_NOTEBOOK.md` for the complete one-line installation command.

### Running the Hyperparameter Tuning Notebook
Follow the instructions in `RUN_HYPERPARAMETER_TUNING_NOTEBOOK.md`:
1. Install notebook dependencies.
2. Place or copy `chromatography_final_merged_data.csv` alongside the notebook if not already present.
3. Execute:
   ```
   jupyter nbconvert --to notebook --execute hyperparameter_tuning.ipynb --output hyperparameter_tuning-executed.ipynb
   ```
The executed notebook captures data summaries, model search iterations, and evaluation metrics. Adjust the input and output filenames to reuse the command for other notebooks in this repository.

### Synthetic Data Workflow
`synthetic_data/data_cleaning.py` demonstrates how to:
- Read the raw chromatography dataset.
- Apply the `SyntheticDistributionImputer` to fill missing values while preserving distributional properties.
- Export the imputed dataset to `synthetic_data/synthetic_chromatography_final_merged_data.csv`.

To run the script:
```
python3 synthetic_data/data_cleaning.py
```
Ensure required dependencies from `requirements.txt` are installed beforehand.

### Outputs and Reports
- Executed notebooks and summary CSV files (e.g., `hyperparameter_tuning_summary_*.csv`) provide model comparisons and parameter grids.
- Generated synthetic datasets reside in `synthetic_data/` for downstream modeling or validation.

### Contribution Notes
- Keep sensitive data out of version control; place large or private files under `raw_data/`.
- Update `requirements.txt` and the README when adding new dependencies or workflows.
- Prefer absolute paths when scripting to match the environment assumptions captured in existing notebooks.

