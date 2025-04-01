from math import exp
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
            setattr(self, name, "")
            return ""

    Fore = DummyStyle()
    Style = DummyStyle()
    print("Warning: colorama not found. Output will not be colored.", file=sys.stderr)

DEFAULT_TEST_DIR = Path("tests")
DEFAULT_COMMAND = ["uv", "run", "main.py"]
TEST_FILE_EXTENSION = ".pep"
EXPECTED_OUTPUT_EXTENSION = ".expected"
TIMEOUT_SECONDS = 10


def print_header(text: str, level: int = 1) -> None:
    """Prints a colored header to the console."""
    color = Fore.CYAN if level == 1 else Fore.MAGENTA
    line = "=" * 25
    print(f"\n{color}{line} {text} {line}{Style.RESET_ALL}")


def print_line_diff(expected: str, actual: str) -> None:
    """Prints a standard unified diff (line-based)."""
    print(f"{Fore.YELLOW}--- Line Diff (Unified) ---{Style.RESET_ALL}")
    
    # Split into lines and filter out wildcard matches before diffing
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True)
    
    # Create filtered versions for diffing
    filtered_expected = []
    filtered_actual = []
    
    for i, exp_line in enumerate(expected_lines):
        if exp_line.strip() != "???":
            filtered_expected.append(exp_line)
            if i < len(actual_lines):
                filtered_actual.append(actual_lines[i])
    
    diff = difflib.unified_diff(
        filtered_expected,
        filtered_actual,
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    
    diff_lines = list(diff)
    if not diff_lines:
        print(f"{Fore.GREEN}Outputs are identical (line level).{Style.RESET_ALL}")
        return

    for line in diff_lines:
        if line.startswith("+"):
            print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
        elif line.startswith("-"):
            print(f"{Fore.RED}{line}{Style.RESET_ALL}")
        elif line.startswith("^"):
            print(f"{Fore.BLUE}{line}{Style.RESET_ALL}")
        elif line.startswith("@"):
            print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        else:
            print(f" {line}")

def print_char_hunk_diff(expected: str, actual: str | None, context_chars: int = 15) -> None:
    """Prints a character-by-character diff in hunks with context."""
    print(f"{Fore.YELLOW}--- Character Diff (Hunks, Context: {context_chars}) ---{Style.RESET_ALL}")

    # Split into lines and filter out wildcard matches
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True) if actual else []
    
    # Create filtered versions for diffing
    filtered_expected = []
    filtered_actual = []
    
    for i, exp_line in enumerate(expected_lines):
        if exp_line.strip() != "???":
            filtered_expected.append(exp_line)
            if i < len(actual_lines):
                filtered_actual.append(actual_lines[i])
    
    # Join filtered lines back into strings
    filtered_expected_str = ''.join(filtered_expected)
    filtered_actual_str = ''.join(filtered_actual)

    # Use filtered strings for diff
    matcher = difflib.SequenceMatcher(None, filtered_expected_str, filtered_actual_str, autojunk=False)

def run_test_process(
    command: List[str], test_file_path: Path, debug=False
) -> Dict[str, Any]:
    """Runs the test command as a subprocess and captures output."""
    full_command = command + [str(test_file_path)]
    if debug:
        full_command.append("--debug")
    result_data = {
        "stdout": "",
        "stderr": "",
        "returncode": -1,
        "error": None,
        "timeout": False,
    }
    try:
        process = subprocess.run(
            full_command,
            capture_output=True,
            text=True,  # Decodes using default locale encoding
            timeout=TIMEOUT_SECONDS,
            check=False,  # Don't raise exception on non-zero exit
            encoding="utf-8",  # Be explicit about encoding
            errors="replace",  # Handle potential decoding errors
        )
        result_data["stdout"] = process.stdout
        result_data["stderr"] = process.stderr
        result_data["returncode"] = process.returncode
    except subprocess.TimeoutExpired as e:
        result_data["error"] = (
            f"TimeoutExpired: Process exceeded {TIMEOUT_SECONDS} seconds."
        )
        # Decode stdout/stderr safely if they exist on timeout
        result_data["stdout"] = (
            e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        )
        result_data["stderr"] = (
            e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        )
        result_data["timeout"] = True
    except FileNotFoundError:
        result_data["error"] = (
            f"FileNotFoundError: Command '{full_command[0]}' not found. Is it installed and in PATH?"
        )
    except Exception as e:
        result_data["error"] = f"Unexpected runner error: {type(e).__name__}: {e}"

    return result_data


def check_test_result(
    run_result: Dict[str, Any],
    # Expects already normalized expected output
    expected_output: Optional[str],
) -> Tuple[bool, str, Optional[str]]:  # Return actual output for diffing
    """
    Checks the test result against expected output and conditions.

    Returns:
        Tuple[bool, str, Optional[str]]: (success_status, reason_string, actual_stdout_for_diff)
        The actual_stdout_for_diff is normalized but potentially unstripped.
    """
    if run_result["error"]:
        return (
            False,
            f"Runner error: {run_result['error']}",
            run_result["stdout"],
        )  # Return raw output on runner error
    if run_result["timeout"]:
        # Return potentially partial output on timeout, normalized
        actual_on_timeout = (
            run_result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
        )
        return False, "Test timed out", actual_on_timeout
    if run_result["returncode"] != 0:
        # Return normalized output on non-zero exit
        actual_on_error = run_result["stdout"].replace(
            "\r\n", "\n").replace("\r", "\n")
        return (
            False,
            f"Exited with non-zero status code: {run_result['returncode']}",
            actual_on_error,
        )

    # Normalize actual output's line endings (expected is pre-normalized)
    actual_output_normalized = (
        run_result["stdout"].replace("\r\n", "\n").replace("\r", "\n")
    )

    # Case 1: No expected output file (.expected does not exist)
    if expected_output is None:
        if run_result["stderr"]:  # Fail if there's unexpected stderr
            return (
                False,
                "Expected no stderr (no .expected file), but got stderr",
                actual_output_normalized,
            )
        # Pass if return code 0 and no stderr
        return (
            True,
            "Passed (Exit code 0, no stderr, no .expected file)",
            actual_output_normalized,
        )

    # Case 2: Expected output file exists
    # Compare normalized outputs, stripping trailing whitespace ONLY for the boolean check
    actual_lines = actual_output_normalized.strip().split('\n')
    expected_lines = expected_output.strip().split('\n')

    if len(actual_lines) != len(expected_lines):
        return False, "Output line count mismatch", actual_output_normalized

    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if expected_line.strip() != "???" and actual_line.strip() != expected_line.strip():
            return False, "Output mismatch", actual_output_normalized

    # If outputs match after stripping, check for unexpected stderr
    if run_result["stderr"]:
        # Output is correct, but stderr might indicate warnings or non-fatal issues. Flag as failure.
        return (
            False,
            "Output matches, but unexpected stderr produced",
            actual_output_normalized,
        )

    # Passed: return code 0, output matches (ignoring trailing whitespace), no unexpected stderr
    return (
        True,
        "Passed (Exit code 0, output matches expected)",
        actual_output_normalized,
    )


def main() -> None:
    """Main function to parse arguments and run the test suite."""
    parser = argparse.ArgumentParser(description="Run Pepper language tests.")
    parser.add_argument(
        "test_filter",
        nargs="?",
        help='Optional: Run only tests whose filename contains this string (e.g., "01", "variables").',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print test file contents and expected output.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help=f"Directory containing test files (default: {DEFAULT_TEST_DIR}).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Pass --debug flag to the command being run.",
    )
    parser.add_argument(
        "--command",
        nargs="+",
        default=DEFAULT_COMMAND,
        help=f'Command to run the interpreter/compiler (default: {" ".join(DEFAULT_COMMAND)}).',
    )
    parser.add_argument(
        "--diff",
        choices=["line", "hunk", "both"],  # Changed 'char' to 'hunk'
        default="hunk",  # Default to hunk-based character diff
        help="Type of diff to show on mismatch (default: hunk).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=15,  # Default context characters for hunk diff
        help="Number of context characters for hunk diff (default: 15).",
    )
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        print(
            f"{Fore.RED}Error: Test directory '{args.test_dir}' not found.{Style.RESET_ALL}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not shutil.which(args.command[0]):
        print(
            f"{Fore.YELLOW}Warning: Command '{args.command[0]}' not found in PATH. "
            f"The test runner might fail.{Style.RESET_ALL}",
            file=sys.stderr,
        )

    all_test_files = sorted(args.test_dir.glob(f"*{TEST_FILE_EXTENSION}"))

    if not all_test_files:
        print(
            f"{Fore.YELLOW}No test files (*{TEST_FILE_EXTENSION}) found in '{args.test_dir}'.{Style.RESET_ALL}"
        )
        sys.exit(0)

    if args.test_filter:
        test_files_to_run = [
            f for f in all_test_files if args.test_filter in f.name]
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
    print(f"Diff type on mismatch: {args.diff}")
    if "hunk" in args.diff or "both" in args.diff:
        print(f"Hunk diff context chars: {args.context}")
    print(f"Running {total_tests} test(s)...")

    for test_file_path in test_files_to_run:
        test_name = test_file_path.name
        print_header(f"Running: {test_name}", level=2)

        expected_output_path = test_file_path.with_suffix(
            EXPECTED_OUTPUT_EXTENSION)
        expected_output_content: Optional[str] = None  # Raw content from file
        normalized_expected_output: Optional[str] = (
            None  # Normalized for comparison/diff
        )

        try:
            # Read test file content
            test_content = test_file_path.read_text(encoding="utf-8")
            if args.verbose:
                print(
                    f"{Fore.YELLOW}--- Test Contents ({test_name}) ---{Style.RESET_ALL}"
                )
                print(test_content.strip())  # Show stripped for brevity
                print(
                    f"{Fore.YELLOW}------------------------------------{Style.RESET_ALL}"
                )

            # Read and normalize expected output file content if it exists
            if expected_output_path.is_file():
                expected_output_content = expected_output_path.read_text(
                    encoding="utf-8"
                )
                # Normalize line endings immediately after reading
                normalized_expected_output = expected_output_content
                if expected_output_content:
                    normalized_expected_output = expected_output_content.replace(
                        "\r\n", "\n"
                    ).replace("\r", "\n")

                if args.verbose:
                    print(
                        f"{Fore.YELLOW}--- Expected Output ({expected_output_path.name}) ---{Style.RESET_ALL}"
                    )
                    # Show stripped normalized version for brevity in verbose mode
                    if normalized_expected_output:
                        print(normalized_expected_output.strip())
                    print(
                        f"{Fore.YELLOW}-----------------------------------------{Style.RESET_ALL}"
                    )
            elif args.verbose:
                print(
                    f"{Fore.YELLOW}--- Expected Output: (None - {expected_output_path.name} not found) ---{Style.RESET_ALL}"
                )

        except Exception as e:
            print(
                f"{Fore.RED}Error reading test or expected file: {e}{Style.RESET_ALL}"
            )
            failed_tests_info.append((test_name, f"File reading error: {e}"))
            continue  # Skip to the next test

        # Run the actual test process
        run_result = run_test_process(args.command, test_file_path, args.debug)

        # Check the results using the normalized expected output
        # actual_output_for_diff will be normalized but potentially unstripped
        success, reason, actual_output_for_diff = check_test_result(
            run_result, normalized_expected_output
        )

        print(f"\n{Fore.YELLOW}--- Result ---{Style.RESET_ALL}")

        if success:
            print(f"{Fore.GREEN}✓ PASSED{Style.RESET_ALL} ({reason})")
            passed_tests += 1
        else:
            print(f"{Fore.RED}✗ FAILED{Style.RESET_ALL} ({reason})")
            failed_tests_info.append((test_name, reason))

            # Display diffs ONLY if failure was due to output mismatch AND there was expected output
            if reason == "Output mismatch" and normalized_expected_output is not None:
                # Use normalized outputs for diffing.
                # Line diff often works better visually with stripped content.
                # Hunk diff can show context including surrounding whitespace.
                expected_for_diff = normalized_expected_output
                # actual_output_for_diff is already normalized from check_test_result

                if args.diff == "line" or args.diff == "both":
                    # Provide stripped versions to unified line diff
                    print_line_diff(
                        expected_for_diff.strip(),
                        (
                            actual_output_for_diff.strip()
                            if actual_output_for_diff
                            else ""
                        ),
                    )
                if args.diff == "hunk" or args.diff == "both":
                    # Provide unstripped (but normalized) versions to hunk diff
                    print_char_hunk_diff(
                        expected_for_diff, actual_output_for_diff, args.context
                    )

            # Show stdout only if it wasn't already shown via diffs or if it's relevant to non-mismatch errors
            if reason != "Output mismatch" and run_result["stdout"]:
                print(f"\n{Fore.YELLOW}--- Stdout ---{Style.RESET_ALL}")
                # Show stripped actual stdout
                print(run_result["stdout"].strip())

            # Always show stderr on failure if present
            if run_result["stderr"]:
                print(f"\n{Fore.YELLOW}--- Stderr ---{Style.RESET_ALL}")
                print(
                    f"{Fore.RED}{run_result['stderr'].strip()}{Style.RESET_ALL}"
                )  # Show stripped stderr

    # --- Test Suite Summary ---
    print_header("TEST SUMMARY", level=1)
    failed_count = total_tests - passed_tests
    print(f"Total tests run: {total_tests}")
    print(f"{Fore.GREEN}Passed: {passed_tests}{Style.RESET_ALL}")
    print(
        f"{Fore.RED if failed_count > 0 else Style.RESET_ALL}Failed: {failed_count}{Style.RESET_ALL}"
    )

    if failed_tests_info:
        print(f"\n{Fore.RED}--- Failed Tests Summary ---{Style.RESET_ALL}")
        for name, reason in failed_tests_info:
            print(f"- {name}: {reason}")
        print(f"\n{Fore.RED}Some tests failed.{Style.RESET_ALL}")
        sys.exit(1)  # Exit with non-zero status code for CI/automation
    else:
        print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
        sys.exit(0)  # Exit with zero status code for success


if __name__ == "__main__":
    main()
