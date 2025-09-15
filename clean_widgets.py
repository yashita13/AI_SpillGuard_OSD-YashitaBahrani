from nbformat import read, write, NO_CONVERT
import os

folder = "YashitaBahrani"
notebook_name = "project_oil_spill_detection.ipynb"
filepath = os.path.join(folder, notebook_name)

with open(filepath, "r", encoding="utf-8") as f:
    nb = read(f, as_version=NO_CONVERT)

# Remove all widget metadata and outputs (optional: keep outputs if needed)
for cell in nb.cells:
    cell.metadata.pop('widgets', None)  # remove broken widgets
    # If outputs are causing issues, uncomment next line
    # cell.outputs = []

# Save cleaned notebook
with open(filepath, "w", encoding="utf-8") as f:
    write(nb, f)

print(f"Notebook {notebook_name} cleaned successfully.")
