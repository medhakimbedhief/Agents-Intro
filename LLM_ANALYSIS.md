# German Legal LLM Analysis for Risk Assessment Agent

## Executive Summary

This document provides recommendations for selecting and implementing German-speaking LLMs for legal document risk assessment, specifically for Verwaltungsakt (administrative decisions) analysis.

## Recommended LLM Options

### 1. **Open Source Models (Local/API)**

#### A. **LeoLM (Leo-7B-German-V2)**
- **Provider**: LAION
- **Size**: 7B parameters
- **Strengths**: 
  - Excellent German language capabilities
  - Legal domain fine-tuning available
  - Can run locally with 16GB RAM
  - Good reasoning capabilities
- **Use Case**: Primary recommendation for local deployment
- **Implementation**: Hugging Face Transformers

#### B. **German BERT Legal (DBLP-legal-bert-base-german)**
- **Provider**: DBLP
- **Size**: Base model (110M parameters)
- **Strengths**:
  - Specifically trained on German legal texts
  - Lightweight and fast
  - Excellent for classification tasks
- **Use Case**: Risk classification and pattern detection
- **Implementation**: Sentence Transformers

#### C. **Mistral-7B-German**
- **Provider**: Mistral AI
- **Size**: 7B parameters
- **Strengths**:
  - Strong reasoning capabilities
  - Good German support
  - Open source
- **Use Case**: Complex legal reasoning tasks

### 2. **API-Based Models (Production)**

#### A. **Claude 3.5 Sonnet (Anthropic)**
- **Provider**: Anthropic
- **Strengths**:
  - Excellent reasoning and analysis
  - Strong German language support
  - Explainable outputs
  - Legal domain expertise
- **Use Case**: Production deployment
- **Cost**: ~$3-5 per 1M tokens

#### B. **GPT-4 (OpenAI)**
- **Provider**: OpenAI
- **Strengths**:
  - Strong reasoning capabilities
  - Good German support
  - Extensive legal knowledge
- **Use Case**: High-accuracy production use
- **Cost**: ~$30 per 1M tokens

## Implementation Strategy

### Phase 1: Local Testing (Recommended)
```python
# LeoLM Implementation
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_leolm():
    model_name = "LeoLM/leo-7b-german-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer
```

### Phase 2: RAG Integration
```python
# German Legal BERT for embeddings
from sentence_transformers import SentenceTransformer

def setup_legal_embeddings():
    model = SentenceTransformer('dblp/legal-bert-base-german')
    return model
```

### Phase 3: Production API
```python
# Claude API integration
from anthropic import Anthropic

def setup_claude():
    client = Anthropic(api_key="your-key")
    return client
```

## Risk Detection Prompts

### Template for German Legal Risk Assessment:
```python
LEGAL_RISK_PROMPT = """
Du bist ein erfahrener Rechtsanwalt, der Verwaltungsakte auf rechtliche Risiken prüft.

Analysiere den folgenden Verwaltungsakt und identifiziere:

1. **Unerkanntes Ermessen**: Unbegründete Ermessensentscheidungen
2. **Widersprüche**: Inkonsistenzen mit aktueller Rechtslage
3. **Veraltete Rechtsprechung**: Überholte rechtliche Grundlagen
4. **Unvollständige Angaben**: Fehlende erforderliche Informationen

Dokument: {document_text}

Gib deine Analyse in folgendem JSON-Format zurück:
{
    "risks": [
        {
            "type": "unerkanntes_ermessen|widersprueche|veraltete_rechtsprechung|unvollstaendige_angaben",
            "severity": "niedrig|mittel|hoch",
            "section": "§ X",
            "description": "Beschreibung des Risikos",
            "suggestion": "Verbesserungsvorschlag"
        }
    ]
}
"""
```

## Performance Comparison

| Model | German Legal | Reasoning | Speed | Cost | Local |
|-------|-------------|-----------|-------|------|-------|
| LeoLM | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Free | ✅ |
| German BERT | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Free | ✅ |
| Claude 3.5 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $$$ | ❌ |
| GPT-4 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $$$$ | ❌ |

## Recommended Architecture

### Hybrid Approach:
1. **Local LeoLM** for initial risk detection
2. **German BERT** for embeddings and RAG
3. **Claude API** for final validation and complex reasoning

### Implementation Steps:
1. Set up local LeoLM for testing
2. Implement RAG with German BERT embeddings
3. Create evaluation dataset with our simulated documents
4. Fine-tune prompts based on results
5. Scale to production with Claude API

## Cost Analysis

### Local Deployment:
- LeoLM: $0 (hardware costs only)
- German BERT: $0
- Total: ~$50-100/month for cloud GPU

### API Deployment:
- Claude 3.5: ~$500-1000/month for 1000 documents
- GPT-4: ~$2000-3000/month for 1000 documents

## Next Steps

1. **Immediate**: Test LeoLM with our simulated document
2. **Week 1**: Implement RAG with German BERT
3. **Week 2**: Create evaluation framework
4. **Week 3**: Fine-tune prompts and test with real documents
5. **Week 4**: Deploy production version

## Evaluation Metrics

- **Precision**: % of detected risks that are actual risks
- **Recall**: % of actual risks that were detected
- **F1-Score**: Harmonic mean of precision and recall
- **Processing Time**: Seconds per document
- **Cost per Document**: $ per document processed

## Conclusion

For your use case, I recommend starting with **LeoLM** for local testing and development, then moving to **Claude 3.5** for production. This provides the best balance of cost, performance, and German legal expertise.