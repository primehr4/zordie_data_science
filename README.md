# Resume Intelligence System

A comprehensive system for analyzing resumes, predicting candidate outcomes, and providing actionable recommendations for job seekers and recruiters.

---

## Key Components

### Section Detection & Document Parsing
- Automatically detects and extracts resume sections  
- Supports multiple document formats (PDF, DOCX, TXT)  
- Identifies key information like skills, experience, education, and projects  

### Skill Alignment Analysis
- Matches candidate skills with job requirements  
- Calculates an alignment score (0–100) based on weighted components:
  - Projects: 0–30 points  
  - Work Experience: 0–30 points  
  - Skills: 0–20 points  
  - Education: 0–10 points  
  - Certifications: 0–5 points  
  - Summary: 0–5 points  
- Identifies missing critical skills and provides recommendations  

### Project Validation
- Analyzes project descriptions for technical depth and relevance  
- Verifies projects against GitHub repositories when available  
- Evaluates code quality, complexity (AST-based), and originality  
- Measures test coverage and code quality metrics  
- Detects contribution patterns  

### Resume Formatting Evaluation
- Assesses resume layout, structure, and readability  
- Checks for formatting consistency and professional presentation  
- Provides formatting improvement recommendations  

### Trustworthiness Detection
- Identifies potential exaggerations or inconsistencies  
- Flags copied content and inflated claims  
- Verifies skill claims against evidence in the resume  

### Credential Verification
- Validates educational credentials and certifications  
- Cross-references with LinkedIn and other professional profiles  
- Assesses the credibility of listed achievements  

### Quality Score Engine
- Calculates an overall quality score (0–100) based on weighted components:
  - Skill Alignment (30%)  
  - Project Validation (30%)  
  - Resume Formatting (10%)  
  - Content Trustworthiness (10%)  
  - Credential Verification (10%)  
  - Online Presence (10%)  
- Provides detailed score breakdown and letter grade  
- Normalizes domain-specific scores (GitHub, LeetCode, Kaggle, etc.)  
- Applies customizable weighting based on job requirements  

### Visualization & Reporting
- Generates interactive visualizations of analysis results  
- Creates comprehensive HTML and JSON reports  
- Provides actionable recommendations for improvement  
- SHAP and LIME explanations for model predictions  
- Strengths and weaknesses analysis  

### Link Extraction & Multi-Platform Analysis
- Extracts and validates links from resumes  
- Crawls GitHub, LinkedIn, and other professional platforms  
- Normalizes data from different sources for unified analysis  

### AI-Based Prediction Engine
- Leverages machine learning to predict candidate outcomes:
  - Technical round success probability  
  - Culture-fit assessment  
  - Learning adaptability prediction  
- Feature importance analysis  

### API Layer
- Serves the system via RESTful endpoints:
  - `POST /analyze` – Submit a candidate for analysis  
  - `GET /analysis/{analysis_id}` – Get analysis status or result  
  - `GET /report/{analysis_id}` – Get HTML report for an analysis  
  - `POST /upload-resume` – Upload a resume file  
  - `GET /platforms` – Get supported platforms  
  - `POST /weight-config` – Update weight configuration  

### Workflow Orchestration
- Manages periodic re-scoring and batch predictions  
- Scheduled tasks for candidate re-evaluation  
- Batch processing for multiple candidates  
- Model training and updating  

---

## Getting Started

### Prerequisites

- Python 3.9 or higher

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for text processing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model (required for NLP tasks)
python -m spacy download en_core_web_lg
```

### GitPython

```bash
set GIT_PYTHON_GIT_EXECUTABLE="/usr/bin/git"

echo %GIT_PYTHON_GIT_EXECUTABLE%
```

### Starting the API Server

```bash
python -m resume_intelligence.main api --port 8000
```

### Starting the Workflow Orchestrator

```bash
python -m resume_intelligence.main workflow --output-dir ./workflow_output
```

### Starting the Full System

```bash
python -m resume_intelligence.main full --api-port 8000 --workflow-output-dir ./workflow_output
```

### API Endpoints

- `POST /analyze`: Submit a candidate for analysis
- `GET /analysis/{analysis_id}`: Get analysis status or result
- `GET /report/{analysis_id}`: Get HTML report for an analysis
- `POST /upload-resume`: Upload a resume file
- `GET /platforms`: Get supported platforms
- `POST /weight-config`: Update weight configuration

### To analyze a single resume against a job description:

```bash
python run_analysis.py --resume path/to/resume.pdf --job-description path/to/job.txt [options]
```

Additional options:
- `--github-username`: GitHub username for additional verification
- `--linkedin-url`: LinkedIn URL for additional verification
- `--output-dir`: Directory to store analysis results (default: "output")
- `--verbose`: Enable verbose output

# Resume Intelligence System - End-to-End Pipeline

This document provides an overview of the Resume Intelligence System's end-to-end pipeline for analyzing resumes against job descriptions. The system performs a comprehensive 4-step analysis process to evaluate candidates' resumes and generate detailed reports.

## Pipeline Overview

The Resume Intelligence System pipeline consists of four main steps:

### Step 1: Initial Screening

This step performs the initial analysis of the resume to determine if the candidate meets the minimum requirements for the position.

1. **Document Parsing**: Extract text content from various document formats (PDF, DOCX, TXT)
2. **Section Detection**: Identify and categorize different sections of the resume
3. **Project Validation**: Analyze and validate projects listed in the resume
4. **Skill Alignment**: Calculate an overall alignment score (0-100) based on:
   - Projects (0-30 points)
   - Work Experience (0-30 points)
   - Skills (0-20 points)
   - Education (0-10 points)
   - Certifications (0-5 points)
   - Summary (0-5 points)
5. **Visualization**: Generate visual representations of the analysis results

If the candidate's skill alignment score is below the threshold (65%), the analysis stops here and generates a rejection report.

### Step 2: In-Depth Analysis

If the candidate passes the initial screening, the system performs a more detailed analysis:

1. **Formatting Analysis**: Evaluate the resume's formatting, structure, and readability
2. **Link Analysis**: Extract and analyze external links (GitHub, LinkedIn, portfolio, etc.)
3. **Trustworthiness Analysis**: Assess the credibility and consistency of the information
4. **Credibility Verification**: Verify credentials and claims made in the resume

### Step 3: Calculating Overall Quality Score

Based on all analyses, the system calculates an overall quality score (0-100) with the following component weights:

1. Skill Alignment (30.0%)
2. Project Validation (30.0%)
3. Resume Formatting (10.0%)
4. Content Trustworthiness (10.0%)
5. Credential Verification (10.0%)
6. Online Presence (10.0%)

The system also generates visualizations of the overall quality score and its components.

### Step 4: Generating Comprehensive Report

Finally, the system generates a detailed report that includes:

- Overall quality score and grade
- Component scores and analysis
- Strengths and weaknesses
- Recommendations for improvement
- Visual representations of the analysis

## Output

The system generates the following outputs for each analysis:

1. **Comprehensive Report** (`comprehensive_analysis_report.html`): Detailed report with all analysis results
2. **Analysis Summary** (`analysis_summary.txt`): Text summary of key metrics and findings
3. **Component Results** (JSON files): Detailed results for each analysis component
4. **Visualizations** (PNG files): Visual representations of analysis results

## System Components

### Core Components

- **Document Parser**: Extracts text from various document formats
- **Section Detector**: Identifies and categorizes resume sections
- **Skill Matcher**: Analyzes skill alignment with job requirements
- **Project Validator**: Validates and scores projects listed in the resume
- **Formatting Scorer**: Evaluates resume formatting and structure
- **Trustworthiness Detector**: Assesses content credibility
- **Credibility Engine**: Verifies credentials and claims
- **Quality Score Engine**: Calculates overall quality score
- **Link Extraction System**: Extracts and analyzes external links
- **Visualizer**: Generates visual representations of analysis results
- **Explainability Layer**: Generates comprehensive reports

### Pipeline Implementation

The pipeline is implemented in two main files:

1. **`resume_pipeline.py`**: Implements the complete 4-step analysis pipeline
2. **`run_analysis.py`**: Provides a command-line interface for running analyses
