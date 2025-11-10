## Running `hyperparameter_tuning.ipynb`

Follow these steps from the project root (`/Users/suayshkulashri/Desktop/Kashish Work/Chromatography`):

1. **Install dependencies (first time or when packages change):**
   ```
   python3 -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org matplotlib numpy pandas scipy seaborn nbconvert jupyter
   ```

2. **Ensure the expected CSV is next to the notebook:**
   ```
   cp raw_data/chromatography_final_merged_data.csv .
   ```
   Skip this if the file already exists in the project root.

3. **Execute the notebook and save an executed copy:**
   ```
   jupyter nbconvert --to notebook --execute hyperparameter_tuning.ipynb --output hyperparameter_tuning-executed.ipynb
   ```

`hyperparameter_tuning-executed.ipynb` will contain the outputs produced during execution. Only works when the ipynb file in PWD.

## Reusing These Commands

- **Dependency installation:** works for any notebook that relies on the listed packages. Add/remove packages to match the notebook’s imports.
- **CSV copy step:** specific to notebooks expecting `chromatography_final_merged_data.csv` in the working directory. Adjust as needed for other input files.
- **`nbconvert` execution:** the same pattern runs any Jupyter notebook. Update the input (`--execute <notebook.ipynb>`) and output (`--output <executed-copy.ipynb>`) filenames to match the notebook you want to run.

