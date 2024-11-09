# Run the FormatDataForModelling notebook automatically for different buffer sizes
# Additionally creates Temporary notebooks that can be deleted afterwards
# Note that you should FormatDataForModelling.ipynb at least once so the cache
# that links the sensors to census data can be creatd.
import papermill as pm
import os

BUFFER_SIZES = [50, 100, 200, 400, 500, 600, 1000]
INPUT_NOTEBOOK = "FormatDataForModelling.ipynb"
OUTPUT_DIR = "."

os.makedirs(OUTPUT_DIR, exist_ok=True)

for BUFFER_SIZE in BUFFER_SIZES:
    OUTPUT_NOTEBOOK = f"{OUTPUT_DIR}/Temp_Notebook_BufferSize_{BUFFER_SIZE}.ipynb"
    print(f"Running for buffer size: {BUFFER_SIZE}: {OUTPUT_NOTEBOOK}")
    pm.execute_notebook(
        INPUT_NOTEBOOK,
        OUTPUT_NOTEBOOK,
        parameters={"buffer_size_m": BUFFER_SIZE}
    )

# TODO open the csv files and check they have the same rows and columns
# f"formatted_data_for_modelling_allsensors_{buffer_size}_outlierremovaleachsensor.csv"