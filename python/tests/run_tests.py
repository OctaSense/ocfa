"""
Test runner script

Runs all unit tests and generates coverage report.
"""

import sys
import pytest


def main():
    """Run all tests"""
    args = [
        'tests/',
        '-v',                    # Verbose
        '--tb=short',           # Short traceback
        '--color=yes',          # Colored output
        '--cov=ocfa',           # Coverage for ocfa package
        '--cov-report=term-missing',  # Show missing lines
        '--cov-report=html',    # Generate HTML report
    ]

    # Run pytest
    exit_code = pytest.main(args)

    if exit_code == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ Tests failed with exit code {exit_code}")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
