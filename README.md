# Exam Grader CV

## Overview
This project is an automatic exam grading system that evaluates student responses and gives feedback. The input is a directory containing answered pages of one exam, and ouputs another directory containing the correction for each page of the exam.

## Usage

1. Install dependencies: `pip install -r requirements.txt`.
2. Prepare your exam metadata in a json file. An example json is present in the directory exams/
3. Run `python generator.py {name_of_json_file}`. 
4. Fill out the exam and prepare the photos under a directory.
3. Run the grader: `python grader.py {path_to_directory}/`. The grader will grade the pages ordered by the filename.
4. View results in the output directory: graded_exam

## CLI

To use the generator, simply pass the json file path to the CLI program, as described before: `python generator.py {path_to_exam_json}`.

The grader accepts one additional parameter, `--dpi`, to force a specific DPI for your input images. The DPI is an important factor in our algorithm, thus it relies on a precise DPI information. In the case where your images metadata were corrupted for some reason, you may enforce a DPI value by running the code as `python grader.py {path_to_exam_images_dir}/ --dpi 300` (note the `/` character after the path, it is needed to specify that it's a directory).

Both commands have a `-h` parameter to show a help message, you may use it if needed.

## File Structure

| File | Purpose |
|------|---------|
| `grader.py` | Entry point for the grading system |
| `generator.py` | Generate standard exam from json |
| `preprocess.py` | Preprocess the input image (page detection) |
| `rectify.py` | Dectect corners and rectify the image (standardize) |
| `process_question.py` | Parse and recognize the questions |
| `exam.py` | Exam type definitions |
| `util.py` | Utility functions |
| `exams/*` | Sample exams json (try `exam_E01.json`!) |


