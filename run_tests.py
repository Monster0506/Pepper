import os
import subprocess
import argparse
from colorama import init, Fore, Style

# Initialize colorama for Windows
init()


def run_test(test_file):
    """Run a single test file and return its output and success status."""
    try:
        result = subprocess.run(['uv', 'run', 'main.py', test_file],
                                capture_output=True,
                                text=True,
                                check=True,
                                timeout=10,
                                )
        return result.stdout, result.returncode == 0
    except Exception as e:
        return str(e), False


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{'='*20} {text} {'='*20}{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(description='Run Pepper language tests')
    parser.add_argument('test_number', nargs='?', type=int,
                        help='Specific test number to run (e.g., 1 for 01_variables.pep)')
    args = parser.parse_args()

    # Get all test files
    test_files = sorted([f for f in os.listdir('tests') if f.endswith('.pep')])

    if args.test_number:
        # Run specific test
        test_num = str(args.test_number).zfill(2)
        matching_tests = [
            f for f in test_files if f.startswith(f'{test_num}_')]
        if not matching_tests:
            print(
                f"{Fore.RED}No test file found for test number {args.test_number}{Style.RESET_ALL}")
            return
        test_files = matching_tests

    total_tests = len(test_files)
    passed_tests = 0

    print_header("PEPPER LANGUAGE TEST SUITE")

    for test_file in test_files:
        test_path = os.path.join('tests', test_file)
        print_header(f"Running {test_file}")

        # Print test file contents
        print(f"{Fore.YELLOW}Test Contents:{Style.RESET_ALL}")
        with open(test_path, 'r') as f:
            print(f.read())

        # Run test
        output, success = run_test(test_path)

        # Print results
        print(f"\n{Fore.YELLOW}Test Output:{Style.RESET_ALL}")
        print(output)

        if success:
            print(f"{Fore.GREEN}✓ Test passed{Style.RESET_ALL}")
            passed_tests += 1
        else:
            print(f"{Fore.RED}✗ Test failed{Style.RESET_ALL}")

    # Print summary
    print_header("TEST SUMMARY")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    if passed_tests == total_tests:
        print(f"{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Some tests failed.{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
