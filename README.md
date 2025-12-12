# Exam Grader CV

## Overview
This project is an exam grading system that evaluates student responses and gives feedback.

## Usage

### Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your exam metadata in a json file
3. Run `python generator.py {name_of_json_file}`. 
4. Fill out the exam and prepare the photos under a directory
3. Run the grader: `python grader.py {path_to_directory}`. The grader will grade the pages ordered by the filename.
4. View results in the output directory: graded_exam


## File Structure

| File | Purpose |
|------|---------|
| `grader.py` | Entry point for the grading system |
| `generator.py` | Generate standard exam |
| `preprocess.py` | Preprocess the input image |
| `rectify.py` | Dectect corners and rectify the image (standardize) |
| `process_question.json` | Parse and recognize the questions |
| `exam.py` | Exam class |
| `util.py` | Utility functions |
| `exams/*` | Exam metadata in json |


