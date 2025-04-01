
srcFolder = r'\.'  # Path to the folder containing Jupyter notebooks
desFolder = r'\.'

import os
import nbformat
from nbconvert import PythonExporter

def convertNotebook(notebookPath, modulePath):
    with open(notebookPath) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    with open(modulePath, 'w+') as fh:
        fh.writelines(source)

# For folder creation if doesn't exist
if not os.path.exists(desFolder):
    os.makedirs(desFolder)

for file in os.listdir(srcFolder):
    if os.path.isdir(srcFolder + '\\' + file):
        continue
    if ".ipynb" in file:
        convertNotebook(srcFolder + '\\' + file, desFolder + '\\' + file[:-5] + "py")