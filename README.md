# Legal Risk Assessment Agent for German Administrative Documents

## Overview

This project implements an AI agent for detecting legal risks in German administrative documents (Verwaltungsakte) using LangGraph and ReAct methodology. The agent identifies various types of legal risks including hidden discretion, contradictions, outdated legal precedent, and incomplete information.

## Project Structure

```
├── simulated_legal_document.txt    # Test document with embedded risks
├── risk_labels.json               # Ground truth risk annotations
├── legal_risk_agent.py            # Main agent implementation
├── test_agent.py                  # Test suite
├── requirements.txt               # Dependencies
├── LLM_ANALYSIS.md               # LLM recommendations
└── README.md                     # This file
```

## Features

### Risk Types Detected
- **Unerkanntes Ermessen**: Hidden discretion without proper justification
- **Widersprüche**: Contradictions with current legal framework
- **Veraltete Rechtsprechung**: Outdated legal precedent
- **Unvollständige Angaben**: Missing required information

### Architecture
- **LangGraph**: ReAct methodology implementation
- **Document Chunking**: Legal section-based chunking
- **Pattern Detection**: Regex-based risk identification
- **RAG Ready**: Vector database integration prepared

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Agent
```bash
python legal_risk_agent.py
```

### 3. Run Tests
```bash
python test_agent.py
```

## Simulated Document

The `simulated_legal_document.txt` contains a German administrative decision (Baugenehmigung) with 7 embedded risks:

1. **Height restriction contradiction** (§ 2)
2. **Unusually short validity period** (§ 4)
3. **Unjustified immediate enforceability** (§ 5)
4. **Non-required rainwater infiltration** (§ 7)
5. **Incorrect fee calculation** (§ 8)
6. **Incomplete building plans** (Appendix)
7. **Potentially outdated appeal period** (§ 6)

## LLM Recommendations

### For Local Testing
- **LeoLM (Leo-7B-German-V2)**: Best open-source German legal model
- **German BERT Legal**: For embeddings and classification

### For Production
- **Claude 3.5 Sonnet**: Best balance of cost and performance
- **GPT-4**: Highest accuracy but expensive

See `LLM_ANALYSIS.md` for detailed comparison and implementation strategies.

## Implementation Details

### Chunking Strategy
The agent uses rule-based chunking with German legal patterns:
- `§ \d+ [A-ZÄÖÜ\s]+` for sections
- `[A-ZÄÖÜ\s]{3,}` for headers
- `Abschnitt \d+` for subsections

### Risk Detection
Pattern-based detection using regex patterns for each risk type, with severity assessment and suggestion generation.

### LangGraph Workflow
1. **Chunk Document**: Split into legal sections
2. **Detect Risks**: Apply pattern matching
3. **Analyze Risks**: Categorize and assess severity
4. **Generate Report**: Create structured output

## Output Format

```json
{
  "document_id": "VA-2024-001",
  "total_risks": 7,
  "severity_distribution": {
    "hoch": 3,
    "mittel": 3,
    "niedrig": 1
  },
  "risks": [
    {
      "id": "risk_001",
      "section": "§ 2 BAUVORSCHRIFTEN",
      "risk_type": "widersprueche",
      "severity": "hoch",
      "description": "Höhenbeschränkung widerspricht aktueller Bauordnung",
      "suggestion": "Höhenbeschränkung auf 10 Meter anpassen"
    }
  ]
}
```

## Next Steps

1. **Integrate LeoLM**: Add German LLM for advanced reasoning
2. **Implement RAG**: Add vector database for legal precedent retrieval
3. **Fine-tune Prompts**: Optimize based on test results
4. **Scale to Production**: Deploy with Claude API
5. **Add UI**: Create frontend for human review

## Evaluation

The test suite compares agent output with ground truth annotations, calculating:
- Precision: % of detected risks that are actual risks
- Recall: % of actual risks that were detected
- Processing time and cost metrics

## Contributing

1. Add new risk patterns to `RiskDetectionAgent`
2. Extend chunking patterns in `LegalDocumentChunker`
3. Improve LLM prompts in `LLM_ANALYSIS.md`
4. Add more test documents

## License

This project is for educational and research purposes in legal AI applications.
