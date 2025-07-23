"""
Legal Risk Assessment Agent using LangGraph and ReAct Methodology
For German Legal Documents (Verwaltungsakt)
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver

# For vector storage (RAG)
import chromadb
from chromadb.config import Settings

# For embeddings (you can replace with your preferred model)
from sentence_transformers import SentenceTransformer

class RiskType(Enum):
    UNERKANNTES_ERMESSEN = "unerkanntes_ermessen"
    WIDERSPRUECHE = "widersprueche"
    VERALTETE_RECHTSPRECHUNG = "veraltete_rechtsprechung"
    UNVOLLSTAENDIGE_ANGABEN = "unvollstaendige_angaben"

class Severity(Enum):
    NIEDRIG = "niedrig"
    MITTEL = "mittel"
    HOCH = "hoch"

@dataclass
class Risk:
    id: str
    section: str
    risk_type: RiskType
    severity: Severity
    description: str
    line_reference: str
    suggestion: str

@dataclass
class DocumentChunk:
    content: str
    section: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentState:
    document_text: str
    chunks: List[DocumentChunk]
    identified_risks: List[Risk]
    current_step: str
    reasoning: List[str]
    final_output: Optional[Dict] = None

class LegalDocumentChunker:
    """Chunks legal documents by sections and headings"""
    
    def __init__(self):
        # German legal section patterns
        self.section_patterns = [
            r'§\s*\d+\s+[A-ZÄÖÜ\s]+',  # § 1 GENEHMIGUNG
            r'[A-ZÄÖÜ\s]{3,}',         # BESCHIED, ANHANG
            r'Abschnitt\s+\d+',        # Abschnitt 1
            r'Artikel\s+\d+',          # Artikel 1
        ]
    
    def chunk_document(self, text: str) -> List[DocumentChunk]:
        """Split document into logical chunks"""
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_section = "Header"
        
        for line in lines:
            # Check if line is a section header
            is_header = any(re.match(pattern, line.strip()) for pattern in self.section_patterns)
            
            if is_header and current_chunk:
                # Save previous chunk
                chunks.append(DocumentChunk(
                    content='\n'.join(current_chunk),
                    section=current_section,
                    metadata={'type': 'section'}
                ))
                current_chunk = []
                current_section = line.strip()
            
            current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                content='\n'.join(current_chunk),
                section=current_section,
                metadata={'type': 'section'}
            ))
        
        return chunks

class RiskDetectionAgent:
    """Main agent for detecting legal risks"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.chunker = LegalDocumentChunker()
        
        # Risk detection patterns
        self.risk_patterns = {
            RiskType.WIDERSPRUECHE: [
                r'widerspricht.*aktuell',
                r'könnte.*verstoßen',
                r'entspricht.*nicht',
                r'veraltet'
            ],
            RiskType.UNERKANNTES_ERMESSEN: [
                r'nicht begründet',
                r'willkürlich',
                r'ohne Begründung',
                r'unzulässige Erschwerung'
            ],
            RiskType.UNVOLLSTAENDIGE_ANGABEN: [
                r'fehlen.*Angaben',
                r'unvollständig',
                r'nicht vollständig'
            ]
        }
    
    def analyze_chunk(self, chunk: DocumentChunk) -> List[Risk]:
        """Analyze a single chunk for risks"""
        risks = []
        
        # Pattern-based detection
        for risk_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, chunk.content, re.IGNORECASE)
                for match in matches:
                    risk = Risk(
                        id=f"risk_{len(risks)+1:03d}",
                        section=chunk.section,
                        risk_type=risk_type,
                        severity=self._assess_severity(risk_type, match.group()),
                        description=match.group(),
                        line_reference=f"Section: {chunk.section}",
                        suggestion=self._generate_suggestion(risk_type, match.group())
                    )
                    risks.append(risk)
        
        return risks
    
    def _assess_severity(self, risk_type: RiskType, description: str) -> Severity:
        """Assess risk severity based on type and content"""
        if risk_type == RiskType.WIDERSPRUECHE:
            return Severity.HOCH
        elif risk_type == RiskType.UNERKANNTES_ERMESSEN:
            return Severity.HOCH
        elif risk_type == RiskType.UNVOLLSTAENDIGE_ANGABEN:
            return Severity.MITTEL
        else:
            return Severity.NIEDRIG
    
    def _generate_suggestion(self, risk_type: RiskType, description: str) -> str:
        """Generate suggestions for risk mitigation"""
        suggestions = {
            RiskType.WIDERSPRUECHE: "Aktuelle Rechtslage prüfen und anpassen",
            RiskType.UNERKANNTES_ERMESSEN: "Begründung für Ermessensentscheidung hinzufügen",
            RiskType.UNVOLLSTAENDIGE_ANGABEN: "Fehlende Angaben ergänzen",
            RiskType.VERALTETE_RECHTSPRECHUNG: "Aktuelle Rechtsprechung prüfen"
        }
        return suggestions.get(risk_type, "Allgemeine Überprüfung empfohlen")

def create_legal_risk_graph():
    """Create the LangGraph workflow for legal risk assessment"""
    
    # Define the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("chunk_document", chunk_document_node)
    workflow.add_node("detect_risks", detect_risks_node)
    workflow.add_node("analyze_risks", analyze_risks_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Define edges
    workflow.set_entry_point("chunk_document")
    workflow.add_edge("chunk_document", "detect_risks")
    workflow.add_edge("detect_risks", "analyze_risks")
    workflow.add_edge("analyze_risks", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()

def chunk_document_node(state: AgentState) -> AgentState:
    """Node: Chunk the document into sections"""
    chunker = LegalDocumentChunker()
    state.chunks = chunker.chunk_document(state.document_text)
    state.current_step = "chunking_complete"
    state.reasoning.append(f"Document chunked into {len(state.chunks)} sections")
    return state

def detect_risks_node(state: AgentState) -> AgentState:
    """Node: Detect risks in each chunk"""
    agent = RiskDetectionAgent()
    all_risks = []
    
    for chunk in state.chunks:
        risks = agent.analyze_chunk(chunk)
        all_risks.extend(risks)
    
    state.identified_risks = all_risks
    state.current_step = "risk_detection_complete"
    state.reasoning.append(f"Detected {len(all_risks)} potential risks")
    return state

def analyze_risks_node(state: AgentState) -> AgentState:
    """Node: Analyze and categorize risks"""
    # Group risks by type
    risk_summary = {}
    for risk in state.identified_risks:
        if risk.risk_type.value not in risk_summary:
            risk_summary[risk.risk_type.value] = []
        risk_summary[risk.risk_type.value].append(risk)
    
    state.current_step = "risk_analysis_complete"
    state.reasoning.append(f"Risks categorized: {list(risk_summary.keys())}")
    return state

def generate_report_node(state: AgentState) -> AgentState:
    """Node: Generate final risk report"""
    # Count risks by severity
    severity_counts = {}
    for risk in state.identified_risks:
        if risk.severity.value not in severity_counts:
            severity_counts[risk.severity.value] = 0
        severity_counts[risk.severity.value] += 1
    
    # Create final output
    output = {
        "document_id": "VA-2024-001",  # Extract from document
        "total_risks": len(state.identified_risks),
        "severity_distribution": severity_counts,
        "risks": [
            {
                "id": risk.id,
                "section": risk.section,
                "risk_type": risk.risk_type.value,
                "severity": risk.severity.value,
                "description": risk.description,
                "suggestion": risk.suggestion
            }
            for risk in state.identified_risks
        ],
        "reasoning": state.reasoning
    }
    
    state.final_output = output
    state.current_step = "complete"
    return state

def main():
    """Main function to run the legal risk assessment"""
    
    # Read the simulated document
    with open('simulated_legal_document.txt', 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Initialize state
    initial_state = AgentState(
        document_text=document_text,
        chunks=[],
        identified_risks=[],
        current_step="start",
        reasoning=[],
        final_output=None
    )
    
    # Create and run the workflow
    workflow = create_legal_risk_graph()
    result = workflow.invoke(initial_state)
    
    # Save results
    with open('agent_output.json', 'w', encoding='utf-8') as f:
        json.dump(result.final_output, f, indent=2, ensure_ascii=False)
    
    print("Risk assessment complete!")
    print(f"Total risks detected: {result.final_output['total_risks']}")
    print(f"Severity distribution: {result.final_output['severity_distribution']}")

if __name__ == "__main__":
    main()