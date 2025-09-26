#!/usr/bin/env python3
"""
Script to execute all Jupyter notebooks in dayname directories
and populate them with output.
"""

import os
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook(notebook_path):
    """Execute a single notebook and save the output."""
    print(f"Executing {notebook_path}...")

    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Configure the executor with 10 minute timeout
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})

        # Save the executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"✓ Successfully executed {notebook_path}")
        return True

    except Exception as e:
        print(f"✗ Error executing {notebook_path}: {str(e)}")
        return False

def main():
    """Main function to execute all notebooks in dayname directories."""
    script_dir = Path(__file__).parent

    # Find all directories that contain .ipynb files
    directories = []
    for item in script_dir.iterdir():
        if item.is_dir() and list(item.glob('*.ipynb')):
            directories.append(item.name)

    directories.sort()

    if not directories:
        print("No directories with notebook files found")
        return

    total_notebooks = 0
    successful_executions = 0

    for directory in directories:
        dir_path = script_dir / directory

        # Find all .ipynb files in the directory
        notebook_files = list(dir_path.glob('*.ipynb'))

        print(f"\nProcessing {len(notebook_files)} notebooks in {directory}/")

        for notebook_path in sorted(notebook_files):
            total_notebooks += 1
            if execute_notebook(notebook_path):
                successful_executions += 1

    print(f"\n{'='*50}")
    print(f"Execution complete!")
    print(f"Total notebooks: {total_notebooks}")
    print(f"Successfully executed: {successful_executions}")
    print(f"Failed: {total_notebooks - successful_executions}")

if __name__ == "__main__":
    main()