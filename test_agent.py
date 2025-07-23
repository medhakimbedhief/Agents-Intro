"""
Test script for the Legal Risk Assessment Agent
"""

import json
from legal_risk_agent import create_legal_risk_graph, AgentState, LegalDocumentChunker, RiskDetectionAgent

def test_chunking():
    """Test the document chunking functionality"""
    print("Testing document chunking...")
    
    # Read the simulated document
    with open('simulated_legal_document.txt', 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Test chunking
    chunker = LegalDocumentChunker()
    chunks = chunker.chunk_document(document_text)
    
    print(f"Document chunked into {len(chunks)} sections:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk.section} ({len(chunk.content)} chars)")
    
    return chunks

def test_risk_detection():
    """Test the risk detection functionality"""
    print("\nTesting risk detection...")
    
    # Read the simulated document
    with open('simulated_legal_document.txt', 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Test risk detection
    agent = RiskDetectionAgent()
    chunker = LegalDocumentChunker()
    chunks = chunker.chunk_document(document_text)
    
    all_risks = []
    for chunk in chunks:
        risks = agent.analyze_chunk(chunk)
        all_risks.extend(risks)
    
    print(f"Detected {len(all_risks)} risks:")
    for risk in all_risks:
        print(f"  - {risk.risk_type.value}: {risk.description[:50]}...")
    
    return all_risks

def test_full_workflow():
    """Test the complete LangGraph workflow"""
    print("\nTesting full workflow...")
    
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
    
    print("Workflow completed successfully!")
    print(f"Final output: {result.final_output}")
    
    return result

def compare_with_ground_truth():
    """Compare agent output with ground truth"""
    print("\nComparing with ground truth...")
    
    # Load ground truth
    with open('risk_labels.json', 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    # Run agent
    result = test_full_workflow()
    
    # Compare results
    agent_risks = result.final_output['risks']
    ground_truth_risks = ground_truth['risks']
    
    print(f"Ground truth risks: {len(ground_truth_risks)}")
    print(f"Agent detected risks: {len(agent_risks)}")
    
    # Simple comparison
    detected_types = set(risk['risk_type'] for risk in agent_risks)
    ground_truth_types = set(risk['risk_type'] for risk in ground_truth_risks)
    
    print(f"Ground truth risk types: {ground_truth_types}")
    print(f"Agent detected risk types: {detected_types}")
    
    # Calculate basic metrics
    precision = len(detected_types.intersection(ground_truth_types)) / len(detected_types) if detected_types else 0
    recall = len(detected_types.intersection(ground_truth_types)) / len(ground_truth_types) if ground_truth_types else 0
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

def main():
    """Run all tests"""
    print("=== Legal Risk Assessment Agent Test Suite ===\n")
    
    try:
        # Test individual components
        test_chunking()
        test_risk_detection()
        
        # Test full workflow
        test_full_workflow()
        
        # Compare with ground truth
        compare_with_ground_truth()
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()