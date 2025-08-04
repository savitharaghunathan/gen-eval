#!/usr/bin/env python3
"""
Interactive Demo Script for GenEval Framework
Allows control over number of test cases and shows detailed evaluation output
"""

import sys
import yaml
import json
import logging
from pathlib import Path

# # Configure logging to show INFO level messages
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from geneval import GenEvalFramework, LLMManager

def load_test_data():
    """Load test data from YAML file"""
    test_data_path = project_root / "tests" / "test_data_clean.yaml"
    
    if not test_data_path.exists():
        print(f"Test data file not found: {test_data_path}")
        return None
        
    with open(test_data_path, 'r') as f:
        return yaml.safe_load(f)

def get_user_preferences():
    """Get user preferences for the demo"""
    print("GenEval Framework Interactive Demo")
    print("=" * 50)
    
    # Check if config file exists
    config_path = project_root / "config" / "llm_config.yaml"
    if not config_path.exists():
        print(f"\n LLM configuration file not found: {config_path}")
        print("Please create a config file with your LLM provider settings.")
        print("Example config:")
        print("  providers:")
        print("    openai:")
        print("      enabled: true")
        print("      default: true")
        print("      api_key_env: \"OPENAI_API_KEY\"")
        print("      model: \"gpt-4o-mini\"")
        print("\nExiting demo - please configure your LLM settings first.")
        return None, None, None
    
    print(f"\n✅ Using LLM configuration from: {config_path}")
    print("The framework will use the default provider from your config file.")
    
    # Number of test cases
    while True:
        try:
            num_cases = input(f"\nHow many test cases to run? (1-10, default: 3): ").strip()
            if not num_cases:
                num_cases = 3
                break
            num_cases = int(num_cases)
            if 1 <= num_cases <= 10:
                break
            else:
                print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")
    
    # Metric selection - show unique metrics
    unique_metrics = [
        "context_precision_without_reference",
        "context_precision_with_reference",
        "context_recall", 
        "context_entity_recall",
        "noise_sensitivity",
        "response_relevancy",
        "faithfulness",
        "answer_relevance",
        "context_relevance"
    ]
    
    print(f"\nAvailable metrics (9 unique):")
    for i, metric in enumerate(unique_metrics, 1):
        print(f"{i:2d}. {metric}")
    
    print(f"\nMetric selection:")
    print("- Enter 'all' for all 9 metrics")
    print("- Enter numbers (comma-separated, e.g., 1,3,6)")
    
    while True:
        selection = input("\nSelect metrics (default: all): ").strip().lower()
        if not selection or selection == "all":
            selected_metrics = unique_metrics
            break
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(",")]
                if all(0 <= i < len(unique_metrics) for i in indices):
                    selected_metrics = [unique_metrics[i] for i in indices]
                    break
                else:
                    print("Invalid metric numbers. Please use numbers 1-9.")
            except ValueError:
                print("Please enter 'all' or numbers separated by commas (e.g., 1,3,6)")
    
    return int(num_cases), selected_metrics, "config"

def convert_to_framework_metrics(selected_metrics, test_data):
    """Convert unique metrics to framework-specific format"""
    ragas_available = test_data['framework_config']['metrics']['ragas']
    deepeval_available = test_data['framework_config']['metrics']['deepeval']
    
    framework_metrics = []
    
    for metric in selected_metrics:
        # Add RAGAS version if available
        if metric in ragas_available:
            framework_metrics.append(f"ragas.{metric}")
        
        # Add DeepEval version if available  
        if metric in deepeval_available:
            framework_metrics.append(f"deepeval.{metric}")
    
    return framework_metrics

def display_test_case_info(test_case, case_num, total_cases):
    """Display information about the current test case"""
    print(f"\n{'='*60}")
    print(f"TEST CASE {case_num}/{total_cases}: {test_case['id']}")
    print(f"{'='*60}")
    print(f"Question: {test_case['user_input']}")
    print(f"Response: {test_case['response']}")
    print(f"Reference: {test_case['reference']}")
    print(f"Context Length: {len(test_case['retrieved_contexts'])} characters")

def display_evaluation_results(results):
    """Display evaluation results in JSON format"""
    json_results = {}
    for metric_key, (adapter_name, output) in results.items():
        json_results[metric_key] = {
            "adapter": adapter_name,
            "metrics": []
        }
        
        for metric_result in output.metrics:
            metric_data = {
                "name": metric_result.name,
                "score": metric_result.score,
                "details": metric_result.details if hasattr(metric_result, 'details') else None
            }
            json_results[metric_key]["metrics"].append(metric_data)
        
        if output.metadata:
            json_results[metric_key]["metadata"] = output.metadata
    
    print(json.dumps(json_results, indent=2, ensure_ascii=False))

def calculate_test_case_stats(all_results):
    """Calculate statistics by test case and adapter.metric"""
    adapter_metric_scores = {}
    
    for case_idx, case_results in enumerate(all_results):
        for metric_key, (adapter_name, output) in case_results.items():
            for metric_result in output.metrics:
                # Use adapter.metric format to keep them separate
                full_metric_name = f"{adapter_name}.{metric_result.name}"
                if full_metric_name not in adapter_metric_scores:
                    adapter_metric_scores[full_metric_name] = []
                if metric_result.score is not None:
                    adapter_metric_scores[full_metric_name].append({
                        'case': case_idx + 1,
                        'score': metric_result.score
                    })
    
    return adapter_metric_scores

def display_final_summary(all_results, metrics, num_cases):
    """Display final summary by test case and keep adapters separate"""
    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY - {num_cases} Test Cases")
    print(f"{'='*100}")
    
    adapter_metric_scores = calculate_test_case_stats(all_results)
    
    print(f"{'Adapter.Metric':<40} {'Cases':<8} {'Avg Score':<12} {'Min':<8} {'Max':<8}")
    print("-" * 85)
    
    for full_metric_name, score_data in sorted(adapter_metric_scores.items()):
        if score_data:
            scores = [item['score'] for item in score_data]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"{full_metric_name:<40} {len(scores):<8} {avg_score:<12.3f} {min_score:<8.3f} {max_score:<8.3f}")
        else:
            print(f"{full_metric_name:<40} {'0':<8} {'N/A':<12} {'N/A':<8} {'N/A':<8}")
    
    # Test case summary
    print(f"\n{'='*60}")
    print(f"TEST CASE SUMMARY")
    print(f"{'='*60}")
    
    for case_idx in range(num_cases):
        case_num = case_idx + 1
        print(f"\nTest Case {case_num}:")
        case_metrics = {}
        
        if case_idx < len(all_results):
            case_results = all_results[case_idx]
            for metric_key, (adapter_name, output) in case_results.items():
                for metric_result in output.metrics:
                    full_metric_name = f"{adapter_name}.{metric_result.name}"
                    if metric_result.score is not None:
                        case_metrics[full_metric_name] = metric_result.score
        
        if case_metrics:
            for metric_name, score in sorted(case_metrics.items()):
                print(f"  {metric_name:<35}: {score:.3f}")
        else:
            print("  No results available")

def initialize_llm():
    """Initialize LLM using config file"""
    try:
        print("Initializing LLM from config file...")
        llm_manager = LLMManager()
        if llm_manager.select_provider():
            provider_info = llm_manager.get_llm_info()
            print(f"✅ LLM initialized successfully: {provider_info['provider']} ({provider_info['model']})")
            return llm_manager, provider_info['provider']
        else:
            print("❌ No default LLM provider configured in config file")
            return None, None
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return None, None

def main():
    """Main demo function"""
    # Load test data
    test_data = load_test_data()
    if not test_data:
        return
    
    # Get user preferences
    preferences = get_user_preferences()
    if preferences[0] is None:
        return
    num_cases, selected_metrics, llm_provider = preferences
    
    # Convert unique metrics to framework-specific format
    metrics = convert_to_framework_metrics(selected_metrics, test_data)
    
    print(f"\nConfiguration:")
    print(f"   LLM Provider: Config-driven")
    print(f"   Test cases: {num_cases}")
    print(f"   Selected metrics: {len(selected_metrics)} unique ({', '.join(selected_metrics)})")
    print(f"   Framework evaluations: {len(metrics)} total ({', '.join(metrics)})")
    
    # Initialize LLM
    print(f"\nInitializing LLM...")
    llm_manager, actual_provider = initialize_llm()
    
    if not llm_manager:
        print("❌ LLM initialization failed.")
        print("Please check your config file and API keys.")
        print("Exiting demo - please configure your LLM settings and try again")
        return
    else:
        print(f"✅ Using {actual_provider.upper()} as LLM provider")
    
    # Initialize framework with LLM
    print(f"\nInitializing GenEval Framework...")
    framework = GenEvalFramework(llm_manager=llm_manager)
    
    # Run evaluations
    all_results = []
    
    for i in range(num_cases):
        test_case = test_data['test_cases'][i]
        
        # Display test case info
        display_test_case_info(test_case, i+1, num_cases)
        
        try:
            # Run actual evaluation
            print(f"\nRunning evaluation with {len(metrics)} metrics...")
            results = framework.evaluate(
                question=test_case['user_input'],
                response=test_case['response'],
                reference=test_case['reference'],
                retrieval_context=test_case['retrieved_contexts'],
                metrics=metrics
            )
            
            # Display results
            display_evaluation_results(results)
            all_results.append(results)
            
        except Exception as e:
            print(f"Error evaluating case {i+1}: {e}")
            continue
    
    # Display final summary if we have results
    if all_results:
        display_final_summary(all_results, metrics, num_cases)
    
    print(f"\nDemo completed!")
    print(f"Evaluated {len(all_results)} test cases successfully")

if __name__ == "__main__":
    main() 