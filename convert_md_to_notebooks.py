#!/usr/bin/env python3
"""
Convert markdown files from dayname/source/ to Jupyter notebooks in dayname/
"""

import os
import subprocess
from pathlib import Path

def convert_md_files_for_day(day_name):
    """Convert all markdown files from dayname/source/ to notebooks in dayname/"""
    source_dir = Path(day_name) / "source"
    target_dir = Path(day_name)

    # Check if source directory exists
    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist")
        return 0

    # Find all .md files in source directory
    md_files = list(source_dir.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {source_dir}/")
        return 0

    print(f"Found {len(md_files)} markdown files to convert in {day_name}:")

    converted_count = 0
    for md_file in md_files:
        print(f"Converting {md_file.name}...")

        # Use jupytext to convert the file
        try:
            subprocess.run([
                "jupytext", "--to", "notebook", str(md_file.name)
            ], cwd=str(source_dir), check=True)

            # Move the generated notebook to the parent directory
            notebook_name = md_file.stem + ".ipynb"
            source_notebook = source_dir / notebook_name
            target_notebook = target_dir / notebook_name

            if source_notebook.exists():
                source_notebook.rename(target_notebook)
                print(f"Created {target_notebook}")
                converted_count += 1
            else:
                print(f"Warning: Expected notebook {source_notebook} was not created")

        except subprocess.CalledProcessError as e:
            print(f"Error converting {md_file.name}: {e}")
        except FileNotFoundError:
            print("Error: jupytext not found. Install with: pip install jupytext")
            return converted_count

    return converted_count

def convert_md_files():
    """Convert all markdown files for all dayname directories"""
    total_converted = 0

    # Find all directories that have a 'source' subdirectory
    current_dir = Path(".")
    day_directories = []

    for item in current_dir.iterdir():
        if item.is_dir() and (item / "source").exists():
            day_directories.append(item.name)

    day_directories.sort()

    if not day_directories:
        print("No dayname/source directories found")
        return

    for day_name in day_directories:
        print(f"\n=== Converting {day_name} ===")
        converted = convert_md_files_for_day(day_name)
        total_converted += converted
        if converted > 0:
            print(f"Successfully converted {converted} files for {day_name}!")

    print(f"\nTotal files converted: {total_converted}")

if __name__ == "__main__":
    convert_md_files()