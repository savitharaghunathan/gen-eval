#!/usr/bin/env python3
"""
Interactive Demo Script for GenEval Framework
Demonstrates both profile-based and direct metric evaluation.
"""

import json
from pathlib import Path

import yaml

from geneval import GenEvalFramework
from geneval.profile_manager import ProfileManager

project_root = Path(__file__).parent


def load_test_data():
    test_data_path = project_root / "tests" / "test_data_clean.yaml"
    if not test_data_path.exists():
        print(f"Test data file not found: {test_data_path}")
        return None
    with open(test_data_path) as f:
        return yaml.safe_load(f)


def check_config():
    config_path = project_root / "config" / "llm_config.yaml"
    if not config_path.exists():
        print(f"\nLLM configuration file not found: {config_path}")
        print("Please create a config file with your LLM provider settings.")
        print("See config/llm_config.yaml.example or the README for details.")
        return None
    return str(config_path)


def select_mode():
    print("\nGenEval Framework Interactive Demo")
    print("=" * 50)
    print("\nEvaluation modes:")
    print("  1. Profile-based evaluation (recommended)")
    print("     Use a predefined profile with weighted scoring and pass/fail verdicts")
    print("  2. Direct metric evaluation")
    print("     Run individual metrics across all adapters")

    while True:
        choice = input("\nSelect mode (1 or 2, default: 1): ").strip()
        if not choice or choice == "1":
            return "profile"
        if choice == "2":
            return "direct"
        print("Please enter 1 or 2")


# ---------------------------------------------------------------------------
# Profile-based evaluation
# ---------------------------------------------------------------------------


def select_profile():
    pm = ProfileManager()
    profiles = sorted(pm.list_profiles())

    print("\nAvailable profiles:")
    for i, name in enumerate(profiles, 1):
        profile = pm.get_profile(name)
        desc = profile.get("description", "")
        metrics = ", ".join(profile["metrics"])
        threshold = profile.get("composite_threshold", "N/A")
        print(f"  {i}. {name:<20} {desc}")
        print(f"     metrics: {metrics}")
        print(f"     composite threshold: {threshold}")

    while True:
        choice = input(f"\nSelect profile (1-{len(profiles)}, default: 1): ").strip()
        if not choice:
            return profiles[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                return profiles[idx]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(profiles)}")


def select_num_cases(max_cases):
    while True:
        try:
            n = input(f"\nHow many test cases to run? (1-{max_cases}, default: 3): ").strip()
            if not n:
                return min(3, max_cases)
            n = int(n)
            if 1 <= n <= max_cases:
                return n
            print(f"Please enter a number between 1 and {max_cases}")
        except ValueError:
            print("Please enter a valid number")


def run_profile_evaluation(framework, test_data):
    profile_name = select_profile()
    num_cases = select_num_cases(len(test_data["test_cases"]))

    pm = ProfileManager()
    profile = pm.get_profile(profile_name)

    print("\nConfiguration:")
    print(f"  Profile: {profile_name}")
    print(f"  Metrics: {', '.join(profile['metrics'])}")
    print(f"  Composite threshold: {profile.get('composite_threshold')}")
    print(f"  Test cases: {num_cases}")

    print(f"\n{'='*70}")
    print(f"Running {num_cases} test cases with profile '{profile_name}'")
    print(f"{'='*70}")

    all_results = []

    for i in range(num_cases):
        tc = test_data["test_cases"][i]
        print(f"\n--- Test Case {i+1}/{num_cases}: {tc['id']} ---")
        print(f"  Question: {tc['user_input'][:80]}...")

        try:
            result = framework.evaluate_profile(
                profile=profile_name,
                question=tc["user_input"],
                response=tc["response"],
                reference=tc["reference"],
                retrieval_context=tc["retrieved_contexts"],
            )
            all_results.append(result)

            status = "PASSED" if result.overall_passed else "FAILED"
            print(f"  Composite score: {result.composite_score:.4f} (threshold: {result.composite_threshold})")
            print(f"  Status: {status}")

            for mr in result.metric_results:
                flag = "pass" if mr.passed else "FAIL"
                print(f"    {mr.name:<35} {mr.score:.4f}  (threshold: {mr.threshold}, weight: {mr.weight}) [{flag}]")

        except Exception as e:
            print(f"  Error: {e}")

    if not all_results:
        print("\nNo results to summarize.")
        return

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY - Profile: {profile_name}")
    print(f"{'='*70}")

    passed = sum(1 for r in all_results if r.overall_passed)
    print(f"Pass rate: {passed}/{len(all_results)} ({passed/len(all_results):.0%})")

    all_metric_names = {mr.name for r in all_results for mr in r.metric_results}
    print(f"\n{'Metric':<35} {'Avg Score':<12} {'Threshold':<12}")
    print("-" * 59)
    for mn in sorted(all_metric_names):
        scores = [mr.score for r in all_results for mr in r.metric_results if mr.name == mn]
        threshold = next(mr.threshold for r in all_results for mr in r.metric_results if mr.name == mn)
        avg = sum(scores) / len(scores)
        print(f"{mn:<35} {avg:<12.4f} {threshold:<12.2f}")

    overall = "PASSED" if all(r.overall_passed for r in all_results) else "FAILED"
    failed = len(all_results) - passed
    if overall == "FAILED":
        print(f"\nOverall: FAILED — {failed} of {len(all_results)} test case(s) scored below threshold")
    else:
        print(f"\nOverall: PASSED — all {len(all_results)} test case(s) met threshold")

    # JSON output option
    save = input("\nSave full results to JSON? (y/N): ").strip().lower()
    if save == "y":
        output_path = input("Output file path (default: eval_results.json): ").strip()
        if not output_path:
            output_path = "eval_results.json"
        results_json = [r.model_dump() for r in all_results]
        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)
        print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Direct metric evaluation (original mode)
# ---------------------------------------------------------------------------


def select_direct_metrics(test_data):
    unique_metrics = [
        "context_precision_without_reference",
        "context_precision_with_reference",
        "context_recall",
        "context_entity_recall",
        "noise_sensitivity",
        "answer_relevancy",
        "faithfulness",
        "context_relevance",
        "context_precision",
    ]

    print("\nAvailable metrics (9 unique):")
    for i, metric in enumerate(unique_metrics, 1):
        print(f"  {i:2d}. {metric}")

    print("\nNote: Some metrics are available in both RAGAS and DeepEval.")
    print("Enter 'all' for all metrics, or numbers (comma-separated, e.g., 1,3,6)")

    while True:
        selection = input("\nSelect metrics (default: all): ").strip().lower()
        if not selection or selection == "all":
            return unique_metrics
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            if all(0 <= i < len(unique_metrics) for i in indices):
                return [unique_metrics[i] for i in indices]
            print("Invalid metric numbers.")
        except ValueError:
            print("Please enter 'all' or numbers separated by commas")


def convert_to_framework_metrics(selected_metrics, framework):
    framework_metrics = []
    for metric in selected_metrics:
        for adapter_name, adapter in framework.adapters.items():
            if metric in adapter.supported_metrics:
                framework_metrics.append(f"{adapter_name}.{metric}")
    return framework_metrics


def run_direct_evaluation(framework, test_data):
    selected_metrics = select_direct_metrics(test_data)
    num_cases = select_num_cases(len(test_data["test_cases"]))

    metrics = convert_to_framework_metrics(selected_metrics, framework)

    ragas_metrics = [m for m in metrics if m.startswith("ragas.")]
    deepeval_metrics = [m for m in metrics if m.startswith("deepeval.")]

    print("\nConfiguration:")
    print(f"  Test cases: {num_cases}")
    print(f"  Selected metrics: {len(selected_metrics)} unique")
    print(f"  Framework evaluations: {len(metrics)} total ({len(ragas_metrics)} RAGAS + {len(deepeval_metrics)} DeepEval)")

    all_results = []

    for i in range(num_cases):
        tc = test_data["test_cases"][i]
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}/{num_cases}: {tc['id']}")
        print(f"{'='*60}")
        print(f"Question: {tc['user_input']}")
        print(f"Response: {tc['response']}")
        print(f"Reference: {tc['reference']}")
        print(f"Context Length: {len(tc['retrieved_contexts'])} characters")

        try:
            print(f"\nRunning evaluation with {len(metrics)} metrics...")
            results = framework.evaluate(
                question=tc["user_input"],
                response=tc["response"],
                reference=tc["reference"],
                retrieval_context=tc["retrieved_contexts"],
                metrics=metrics,
            )

            json_results = {}
            for metric_key, (adapter_name, output) in results.items():
                json_results[metric_key] = {"adapter": adapter_name, "metrics": []}
                for metric_result in output.metrics:
                    json_results[metric_key]["metrics"].append(
                        {
                            "name": metric_result.name,
                            "score": metric_result.score,
                            "details": metric_result.details if hasattr(metric_result, "details") else None,
                        }
                    )
                if output.metadata:
                    json_results[metric_key]["metadata"] = output.metadata

            print(json.dumps(json_results, indent=2, ensure_ascii=False))
            all_results.append(results)

        except Exception as e:
            print(f"Error evaluating case {i+1}: {e}")

    if all_results:
        adapter_metric_scores = {}
        for case_results in all_results:
            for _key, (adapter_name, output) in case_results.items():
                for mr in output.metrics:
                    full_name = f"{adapter_name}.{mr.name}"
                    if full_name not in adapter_metric_scores:
                        adapter_metric_scores[full_name] = []
                    if mr.score is not None:
                        adapter_metric_scores[full_name].append(mr.score)

        print(f"\n{'='*85}")
        print(f"FINAL SUMMARY - {num_cases} Test Cases")
        print(f"{'='*85}")
        print(f"{'Adapter.Metric':<40} {'Cases':<8} {'Avg Score':<12} {'Min':<8} {'Max':<8}")
        print("-" * 85)

        for name, scores in sorted(adapter_metric_scores.items()):
            if scores:
                avg = sum(scores) / len(scores)
                print(f"{name:<40} {len(scores):<8} {avg:<12.3f} {min(scores):<8.3f} {max(scores):<8.3f}")

    print(f"\nDemo completed! Evaluated {len(all_results)} test cases successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    test_data = load_test_data()
    if not test_data:
        return

    config_path = check_config()
    if not config_path:
        return

    mode = select_mode()

    print(f"\nUsing LLM configuration from: {config_path}")
    print("Initializing GenEval Framework...")

    try:
        framework = GenEvalFramework(config_path=config_path)
        print("Framework initialized successfully")
    except Exception as e:
        print(f"Framework initialization failed: {e}")
        print("Please check your config file and API keys.")
        return

    try:
        if mode == "profile":
            run_profile_evaluation(framework, test_data)
        else:
            run_direct_evaluation(framework, test_data)
    finally:
        framework.close()


if __name__ == "__main__":
    main()
