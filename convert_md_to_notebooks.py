#!/usr/bin/env python3
"""
Convert markdown files from day_1/source/ and day_2/source/ to Jupyter notebooks
"""

import os
import subprocess
from pathlib import Path

def convert_md_files_for_day(day_num):
    """Convert all markdown files from day_X/source/ to notebooks in day_X/"""
    source_dir = Path(f"day_{day_num}/source")
    target_dir = Path(f"day_{day_num}")

    # Create target directory if it doesn't exist
    target_dir.mkdir(exist_ok=True)

    # Find all .md files in source directory
    md_files = list(source_dir.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in day_{day_num}/source/")
        return 0

    print(f"Found {len(md_files)} markdown files to convert in day_{day_num}:")

    converted_count = 0
    for md_file in md_files:
        print(f"Converting {md_file.name}...")

        # Output notebook path
        notebook_name = md_file.stem + ".ipynb"
        output_path = target_dir / notebook_name

        # Use jupytext to convert
        try:
            subprocess.run([
                "jupytext", "--to", "notebook", str(md_file),
                "--output", str(output_path)
            ], check=True)
            print(f"Created {output_path}")
            converted_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error converting {md_file.name}: {e}")
        except FileNotFoundError:
            print("Error: jupytext not found. Install with: pip install jupytext")
            return converted_count

    return converted_count

def convert_md_files():
    """Convert all markdown files for both day_1 and day_2"""
    total_converted = 0

    for day in [1, 2]:
        print(f"\n=== Converting day_{day} ===")
        converted = convert_md_files_for_day(day)
        total_converted += converted
        if converted > 0:
            print(f"Successfully converted {converted} files for day_{day}!")

    print(f"\nTotal files converted: {total_converted}")

if __name__ == "__main__":
    convert_md_files()