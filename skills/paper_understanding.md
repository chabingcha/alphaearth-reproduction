# Skill: Paper Understanding

## Metadata
- **Name**: paper-understanding
- **Version**: 1.0
- **Description**: Deep analysis of academic papers — extracts hypotheses, methodology, experiments, contributions, and answers analytical questions.
- **Input**: Path to paper PDF or text file
- **Output**: Structured JSON analysis file

## Instructions

You are a paper analysis agent. Your task is to deeply read and analyze an academic paper, extracting structured information that will be used by downstream skills (figure generation and summary writing).

### Step 1: Read the Paper
Read the complete paper text. If provided as PDF, extract text using pymupdf. Read ALL sections — don't skip the appendix, references, or figure captions.

### Step 2: Extract Core Information

**A. Paper Identity**
- Title, authors, affiliations, venue, date
- Research domain and sub-domain
- Keywords

**B. Problem Statement & Motivation**
- What gap does the paper address?
- Why is this problem important?
- What are the limitations of prior work?

**C. Research Questions / Hypotheses**
- List each explicit research question or hypothesis (numbered)
- For each: what would confirm or refute it?

**D. Methodology**
- Overall approach (pipeline / framework)
- For each method used:
  - Name and type (linear, nonlinear, attention-based, etc.)
  - Input data, output data, hyperparameters
  - Why this method was chosen
- Data sources: name, resolution, temporal range, sample size
- Preprocessing steps

**E. Experiments & Validation**
- For each experiment:
  - Goal, setup, metrics, results
- Validation strategy (cross-validation, holdout, etc.)
- Baseline comparisons

**F. Key Results**
- Quantitative results (numbers, tables)
- Qualitative findings
- Claims made based on results

**G. Contributions & Limitations**
- What is new (method, dataset, finding, system)?
- Acknowledged limitations
- Unstated limitations you identify

### Step 3: Output Format

Write the analysis to `results/paper_analysis.json` with this structure:

```json
{
  "paper_identity": {
    "title": "...",
    "authors": "...",
    "year": 2026,
    "domain": "...",
    "keywords": ["..."]
  },
  "problem_statement": {
    "gap": "...",
    "importance": "...",
    "prior_limitations": "..."
  },
  "research_questions": [
    {"id": 1, "question": "...", "validation_method": "...", "result": "..."}
  ],
  "methodology": {
    "overall_approach": "...",
    "methods": [
      {
        "name": "...",
        "type": "linear|nonlinear|attention",
        "input": "...",
        "output": "...",
        "hyperparameters": {},
        "rationale": "..."
      }
    ],
    "data_sources": [
      {"name": "...", "resolution": "...", "temporal_range": "...", "sample_size": "..."}
    ],
    "preprocessing": ["..."]
  },
  "experiments": [
    {"goal": "...", "setup": "...", "metrics": "...", "results": "..."}
  ],
  "key_results": {
    "quantitative": {"...": "..."},
    "qualitative": ["..."],
    "claims": ["..."]
  },
  "contributions": ["..."],
  "limitations": {
    "acknowledged": ["..."],
    "unstated": ["..."]
  }
}
```

### Step 4: Answer Analytical Questions
Also answer these questions in natural language and append to the output:

1. What is the single most important contribution of this paper?
2. How do the three interpretability methods complement each other?
3. Is the validation strategy adequate? What could be improved?
4. How does the LSI system use the interpretability findings?
5. What assumptions does the paper make that may not generalize?
6. If you were to reproduce this paper at a different geographic scale (e.g., a 2°×2° region instead of CONUS), what challenges would you anticipate?

### Quality Criteria
- Every number from the paper must be cited with its page/section
- All 26 environmental variables should be listed with their sources
- The analysis must be detailed enough that someone could reproduce the paper from it alone
- Flag any unclear or ambiguous descriptions in the paper
