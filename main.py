#!/usr/bin/env python3
"""
GenEval - Unified RAG Evaluation Framework

This is a library package designed to be imported, not run directly.
"""

def main():
    print("GenEval - Unified RAG Evaluation Framework")
    print("=" * 50)
    print("This is a library package. Usage:")
    print()
    print("  from geneval import GenEvalFramework, LLMInitializer")
    print()
    print("  # Initialize with LLM provider")
    print("  llm_init = LLMInitializer(provider='openai')")
    print("  framework = GenEvalFramework(llm_initializer=llm_init)")
    print()
    print("  # Evaluate with metrics")
    print("  results = framework.evaluate(")
    print("      question='What is the capital of France?',")
    print("      response='Paris is the capital of France.',")
    print("      reference='Paris',")
    print("      retrieval_context='France is a country in Europe...',")
    print("      metrics=['faithfulness', 'answer_relevance']")
    print("  )")
    print()

if __name__ == "__main__":
    main()
