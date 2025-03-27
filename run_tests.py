import os
import subprocess
import argparse
import sys
import difflib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from colorama import init, Fore, Style
    init()
except ImportError:
    class DummyStyle:
        def __getattr__(self, name: str) -> str:
            return ""
    Fore = DummyStyle()
    Style = DummyStyle()
    print("Warning: colorama not found. Output will not be colored.", file=sys.stderr)

DEFAULT_TEST_DIR = Path('tests')
DEFAULT_COMMAND = ['uv', 'run', 'main.py']
TEST_FILE_EXTENSION = '.pep'
EXPECTED_OUTPUT_EXTENSION = '.expected'
TIMEOUT_SECONDS = 10

def print_header(text: str, level: int = 1) -> None:
    color = Fore.CYAN if level == 1 else Fore.MAGENTA
    line = "=" * 25
    print(f"\n{color}{line} {text} {line}{Style.RESET_ALL}")

def print_diff(expected: str, actual: str) -> None:
    print(f"{Fore.YELLOW}--- Diff ---{Style.RESET_ALL}")
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile='expected',
        tofile='actual',
    )
    for line in diff:
        if line.startswith('+'):
            print(f"{Fore.GREEN}{line}{Style.RESET_ALL}", end='')
        elif line.startswith('-'):
            print(f"{Fore.RED}{line}{Style.RESET_ALL}", end='')
        elif line.startswith('^'):
            print(f"{Fore.BLUE}{line}{Style.RESET_ALL}", end='')
        else:
            print(line, end='')
    print(f"{Fore.YELLOW}------------{Style.RESET_ALL}")


def run_test_process(command: List[str], test_file_path: Path) -> Dict[str, Any]:
    full_command = command + [str(test_file_path)]
    result_data = {
        'stdout': '',
        'stderr': '',
        'returncode': -1,
        'error': None,
        'timeout': False,
    }
    try:
        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        result_data['stdout'] = process.stdout
        result_data['stderr'] = process.stderr
        result_data['returncode'] = process.returncode
    except subprocess.TimeoutExpired as e:
        result_data['error'] = f"TimeoutExpired: Process exceeded {TIMEOUT_SECONDS} seconds."
        result_data['stdout'] = e.stdout or ''
        result_data['stderr'] = e.stderr or ''
        result_data['timeout'] = True
    except FileNotFoundError:
        result_data['error'] = f"FileNotFoundError: Command '{full_command[0]}' not found. Is it installed and in PATH?"
    except Exception as e:
        result_data['error'] = f"Unexpected runner error: {type(e).__name__}: {e}"

    return result_data


def check_test_result(
    test_file: Path,
    run_result: Dict[str, Any],
    expected_output: Optional[str]
) -> Tuple[bool, str]:
    if run_result['error']:
        return False, f"Runner error: {run_result['error']}"
    if run_result['timeout']:
        return False, "Test timed out"
    if run_result['returncode'] != 0:
        return False, f"Exited with non-zero status code: {run_result['returncode']}"

    if expected_output is None:
        if run_result['stderr']:
             return False, f"Expected no stderr, but got stderr (no .expected file found)"
        return True, "Passed (Exit code 0, no stderr, no .expected file)"

    actual_output = run_result['stdout'].strip()
    if actual_output != expected_output:
        return False, "Output mismatch"

    return True, "Passed (Exit code 0, output matches expected)"


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Pepper language tests.')
    parser.add_argument(
        'test_filter',
        nargs='?',
        help='Optional: Run only tests whose filename contains this string (e.g., "01", "variables").'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print test file contents and expected output.'
    )
    parser.add_argument(
        '--test-dir',
        type=Path,
        default=DEFAULT_TEST_DIR,
        help=f'Directory containing test files (default: {DEFAULT_TEST_DIR}).'
    )
    parser.add_argument(
        '--command',
        nargs='+',
        default=DEFAULT_COMMAND,
        help=f'Command to run the interpreter/compiler (default: {" ".join(DEFAULT_COMMAND)}).'
    )
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        print(f"{Fore.RED}Error: Test directory '{args.test_dir}' not found.{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

    if not shutil.which(args.command[0]):
         print(
            f"{Fore.YELLOW}Warning: Command '{args.command[0]}' not found in PATH. "
            f"The test runner might fail.{Style.RESET_ALL}",
            file=sys.stderr
        )

    all_test_files = sorted(args.test_dir.glob(f'*{TEST_FILE_EXTENSION}'))

    if not all_test_files:
        print(f"{Fore.YELLOW}No test files (*{TEST_FILE_EXTENSION}) found in '{args.test_dir}'.{Style.RESET_ALL}")
        sys.exit(0)

    if args.test_filter:
        test_files_to_run = [
            f for f in all_test_files if args.test_filter in f.name
        ]
        if not test_files_to_run:
            print(
                f"{Fore.RED}No test files found matching filter '{args.test_filter}' in '{args.test_dir}'.{Style.RESET_ALL}"
            )
            sys.exit(1)
    else:
        test_files_to_run = all_test_files

    total_tests = len(test_files_to_run)
    passed_tests = 0
    failed_tests_info = []

    print_header("PEPPER LANGUAGE TEST SUITE", level=1)
    print(f"Test directory: {args.test_dir.resolve()}")
    print(f"Interpreter command: {' '.join(args.command)}")
    print(f"Running {total_tests} test(s)...")

    for test_file_path in test_files_to_run:
        test_name = test_file_path.name
        print_header(f"Running: {test_name}", level=2)

        expected_output_path = test_file_path.with_suffix(EXPECTED_OUTPUT_EXTENSION)
        expected_output: Optional[str] = None

        try:
            test_content = test_file_path.read_text()
            if args.verbose:
                print(f"{Fore.YELLOW}--- Test Contents ({test_name}) ---{Style.RESET_ALL}")
                print(test_content.strip())
                print(f"{Fore.YELLOW}------------------------------------{Style.RESET_ALL}")

            if expected_output_path.is_file():
                expected_output = expected_output_path.read_text()
                if args.verbose:
                    print(f"{Fore.YELLOW}--- Expected Output ({expected_output_path.name}) ---{Style.RESET_ALL}")
                    print(expected_output.strip())
                    print(f"{Fore.YELLOW}-----------------------------------------{Style.RESET_ALL}")
            elif args.verbose:
                print(f"{Fore.YELLOW}--- Expected Output: (None - {expected_output_path.name} not found) ---{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error reading test or expected file: {e}{Style.RESET_ALL}")
            failed_tests_info.append((test_name, f"File reading error: {e}"))
            continue

        run_result = run_test_process(args.command, test_file_path)

        success, reason = check_test_result(test_file_path, run_result, expected_output)

        print(f"\n{Fore.YELLOW}--- Result ---{Style.RESET_ALL}")

        if success:
            print(f"{Fore.GREEN}✓ PASSED{Style.RESET_ALL} ({reason})")
            passed_tests += 1
        else:
            print(f"{Fore.RED}✗ FAILED{Style.RESET_ALL} ({reason})")
            failed_tests_info.append((test_name, reason))

            if reason == "Output mismatch" and expected_output is not None:
                 print_diff(expected_output, run_result['stdout'])
            else:
                 if run_result['stdout']:
                      print(f"\n{Fore.YELLOW}--- Stdout ---{Style.RESET_ALL}")
                      print(run_result['stdout'].strip())
                 if run_result['stderr']:
                     print(f"\n{Fore.YELLOW}--- Stderr ---{Style.RESET_ALL}")
                     print(f"{Fore.RED}{run_result['stderr'].strip()}{Style.RESET_ALL}")


    print_header("TEST SUMMARY", level=1)
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if failed_tests_info:
        print(f"\n{Fore.RED}--- Failed Tests ---{Style.RESET_ALL}")
        for name, reason in failed_tests_info:
            print(f"- {name}: {reason}")
        print(f"\n{Fore.RED}Some tests failed.{Style.RESET_ALL}")
        sys.exit(1)
    else:
        print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
        sys.exit(0)


if __name__ == '__main__':
    main()