# Skill: Summary Writing

## Metadata
- **Name**: summary-writing
- **Version**: 1.0
- **Description**: Produces a publication-quality Markdown summary of an academic paper, integrating structured analysis and generated figures.
- **Input**: 
  - `results/paper_analysis.json` (from Skill 1)
  - `results/figures/` directory (from Skill 2)
- **Output**: `results/paper_summary.md`

## Instructions

You are a scientific writing agent. Your task is to write a comprehensive, well-organized Markdown summary of an academic paper. You will receive the structured analysis from Skill 1 and figures from Skill 2.

### Writing Principles

1. **Accuracy**: Every claim must be traceable to the paper. When citing numbers, include them exactly as reported.
2. **Clarity**: Explain complex concepts in plain language. Use examples when helpful.
3. **Structure**: Organize logically — let the paper's own structure guide you, but don't be constrained by it.
4. **Completeness**: Cover methodology, experiments, results, and contributions. Don't skip limitations.
5. **Visual Enhancement**: Embed figures and diagrams where they add value. Every figure needs a descriptive caption.

### Required Sections

#### 1. Paper Metadata
- Title, authors, affiliation, venue, year
- One-sentence TL;DR

#### 2. Problem & Motivation
- What problem does the paper solve?
- Why existing approaches fall short?
- The paper's core insight or hypothesis

#### 3. Methodology
- **3.1 Overall Architecture**: System diagram + explanation
- **3.2 Data Sources**: Table of all data sources with specifications
- **3.3 Interpretability Analysis**: Three methods explained:
  - Spearman Rank Correlation (linear monotonic)
  - Random Forest Regression (nonlinear importance)
  - Multi-Task TabTransformer (attention-based)
- **3.4 Validation Strategy**: Spatial block CV vs random CV, temporal stability
- **3.5 LSI System**: FAISS index, RAG pipeline, LLM integration

#### 4. Key Results
- **4.1 Interpretability Findings**: R² values, top dimension-variable pairs
- **4.2 Spatial & Temporal Robustness**: CV results, stability metrics
- **4.3 LSI System Performance**: LLM-as-Judge evaluation scores
- **4.4 Method Convergence**: How the three methods agree

#### 5. Critical Analysis
- **5.1 Strengths**: What the paper does well
- **5.2 Limitations**: Acknowledged AND unstated limitations
- **5.3 Reproducibility Assessment**: What would be needed to reproduce
- **5.4 Scale Dependence**: How results might change at different geographic scales

#### 6. Contributions & Impact
- Scientific contributions
- Practical implications
- Future research directions

### Output Format

Write the complete summary to `results/paper_summary.md`. Use proper Markdown formatting:

```markdown
# Paper Title

**Authors** | **Venue, Year**

> **TL;DR:** One-sentence summary.

---

## 1. Problem & Motivation
...

## 2. Methodology
### 2.1 Overall Architecture
![System Architecture](figures/fig1_architecture.png)
*Figure 1: ...*

### 2.2 Data Sources
| Source | Variable | Resolution | Temporal | Samples |
|--------|----------|------------|----------|---------|
| ... | ... | ... | ... | ... |

...

## 3. Key Results
...

## 4. Critical Analysis
...

## 5. Contributions & Impact
...
```

### Quality Checklist
- [ ] All numbers cited from the paper are accurate
- [ ] At least 4 figures/diagrams are embedded
- [ ] Data sources table is complete
- [ ] Limitations section includes at least 3 unstated limitations
- [ ] TL;DR is one sentence and captures the essence
- [ ] Methodology section is detailed enough for reproduction
- [ ] Summary is self-contained (understandable without reading the paper)
- [ ] Markdown renders correctly (valid syntax, working image paths)

### CRITICAL
Do NOT write placeholder text like "[Add more details here]" or "[TODO]". Every section must be fully written with actual content from the paper. If the analysis file is missing a detail, infer it from the paper context or note it explicitly as "Not specified in the paper."

The summary should be 2000-4000 words — comprehensive but concise.
