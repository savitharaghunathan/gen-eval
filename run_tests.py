#!/usr/bin/env python3
"""
Test runner script for GenEval framework.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit            # Run only unit tests
    python run_tests.py --coverage        # Run tests with coverage
    python run_tests.py --verbose         # Run tests with verbose output
"""

import subprocess
import sys
import argparse


def run_tests(args):
    """Run pytest with the given arguments"""
    cmd = ["uv", "run", "pytest", "tests/"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=geneval", "--cov-report=term-missing"])
    
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run GenEval tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    exit_code = run_tests(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 