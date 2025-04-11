import re
from typing import List, Dict, Any, Optional
import random
from stdlib import _get_python_type_name
import os
from ExpressionEvaluator import ExpressionEvaluator
from consts import (
    FOR_KEYWORD,
    WHILE_KEYWORD,
    LET_KEYWORD,
    REAS_KEYWORD,
    SHOW_KEYWORD,
    IF_KEYWORD,
    ELIF_KEYWORD,
    ELSE_KEYWORD,
    END_KEYWORD,
    LOOP_END_KEYWORD,
    GOTO_KEYWORD,
    LABEL_KEYWORD,
    RETURN_KEYWORD,
    IMPORT_KEYWORD,
    Function,
    supported_types,
)


class Interpreter:
    # Class attribute for supported types

    def __init__(self, debug=False, debug_show=False):
        self.variables: Dict[str, tuple[Any, str]] = {}  # Global variables
        self.functions: Dict[str, Function] = {}  # Store defined functions
        self.labels: Dict[str, int] = {}  # Label name -> line number (1-based)
        self.script_directory: Optional[str] = (
            None  # Store the directory of the main script
        )
        self.expression_evaluator = ExpressionEvaluator(
            self.variables, self.functions, interpreter=self, debug=debug
        )  # Pass functions dict
        self.debug = debug
        self.debug_show = debug_show
        # No global return_value needed if scope is handled correctly

    def _load_module(self, module_name: str, function_name: str):
        """
        Loads a module and extracts a specific function with extensive debugging.
        """
        if not self.script_directory:
            base_path = "."
            if self.debug:
                print(
                    "  [Module Loader] No script directory set, using current directory."
                )
        else:
            base_path = self.script_directory
            if self.debug:
                print(
                    f"  [Module Loader] Using script directory for module loading: {self.script_directory}"
                )

        module_path = os.path.join(base_path, f"{module_name}.pep")
        if self.debug:
            print(f"  [Module Loader] Attempting to load module: {module_name}")
            print(
                f"  [Module Loader] Looking for module file at resolved path: {module_path}"
            )

        try:
            with open(module_path, "r") as file:
                module_lines = file.readlines()
                if self.debug:
                    print("  [Module Loader] Successfully opened module file.")
                    print(f"  [Module Loader] Module content: {module_lines}")

            # Create a temporary interpreter instance for module loading
            module_interpreter = Interpreter(
                debug=self.debug, debug_show=self.debug_show
            )
            if self.debug:
                print(
                    "  [Module Loader] Creating temporary interpreter instance for module loading."
                )

            module_interpreter.pre_scan(module_lines)
            if self.debug:
                print("  [Module Loader] Pre-scan completed for module.")
                print(
                    f"  [Module Loader] Found functions: {list(module_interpreter.functions.keys())}"
                )

            if function_name not in module_interpreter.functions:
                if self.debug:
                    print(
                        f"  [Module Loader] Function '{function_name}' not found in module '{module_name}'"
                    )
                raise ValueError(
                    f"Function '{function_name}' not found in module '{module_name}'"
                )

            imported_function = module_interpreter.functions[function_name]
            if self.debug:
                print(f"  [Module Loader] Function '{function_name}' found and loaded.")
                print(f"  [Module Loader] Imported function: {imported_function}")
            return imported_function

        except FileNotFoundError:
            if self.debug:
                print(f"  [Module Loader] Module '{module_name}' not found.")
            raise FileNotFoundError(f"Module '{module_name}' not found")
        except ValueError as e:
            if self.debug:
                print(f"  [Module Loader] ValueError: {e}")
            raise ValueError(f"Error loading module '{module_name}': {e}")
        except Exception as e:
            if self.debug:
                print(f"  [Module Loader] Unexpected error: {type(e).__name__}: {e}")
            raise RuntimeError(
                f"Unexpected error loading module '{module_name}': {type(e).__name__}: {e}"
            )

    def execute_line(self, line, line_num_for_error, current_vars):
        """
        Executes a single line in the given variable context.
        Used primarily by REPL or potentially simple sub-execution.
        NOTE: This is a simplified version and won't handle control flow correctly.
              The main execution logic is in `execute_block`.
        """
        evaluator = ExpressionEvaluator(current_vars, self.functions, self, self.debug)
        line = line.strip()
        if not line or line.startswith("%%"):
            return None

        try:
            # --- Seed Command (NEW for REPL) ---
            if line.startswith("?"):  # Quick check before regex
                seed_match = re.match(r"\?\s+(.+)", line)
                if seed_match:
                    seed_value_str = seed_match.group(1).strip()
                    try:
                        seed_value_for_func = int(seed_value_str)
                        random.seed(seed_value_for_func)
                        if self.debug:
                            print(
                                f"  Seeded random generator with int: {seed_value_for_func}"
                            )
                    except ValueError:
                        seed_value_for_func = seed_value_str
                        random.seed(seed_value_for_func)
                        if self.debug:
                            print(
                                f"  Seeded random generator with string: '{seed_value_for_func}'"
                            )
                    return None  # Seed command handled, return nothing
                else:
                    raise ValueError("Invalid seed syntax. Expected '? <seed_value>'")
            elif line.startswith("IMPORT"):
                match = re.match(
                    r"IMPORT\s+([a-zA-Z_]\w*)\s+FROM\s+([a-zA-Z0-9_./\\-]+)", line
                )
                if not match:
                    raise ValueError(
                        "Invalid IMPORT syntax. Expected: IMPORT function_name FROM module_name"
                    )
                function_name, module_name = match.groups()
                try:
                    imported_function = self._load_module(module_name, function_name)
                    self.functions[function_name] = imported_function
                    print(
                        f"Imported function '{function_name}' from module '{module_name}'"
                    )
                    return None
                except (FileNotFoundError, ValueError) as e:
                    raise ValueError(f"Error during import: {e}") from e

            elif line.startswith(LET_KEYWORD):
                return self.handle_let(line, current_vars, evaluator)
            elif line.startswith(REAS_KEYWORD):
                return self.handle_reas(line, current_vars, evaluator)
            elif line.startswith(SHOW_KEYWORD):
                return self.handle_show(line, evaluator)
            elif "|>" in line:  # Function call statement
                return self.handle_function_call(line, evaluator)
            elif (
                "::(" in line
            ):  # Function definition (only declaration in REPL context)
                print("Function defined (syntax checked).")
                # Assume void return for REPL def
                return self.handle_function_declaration(
                    [line, "<-void"], 0, store_function=True
                )
            elif line.startswith(
                (
                    IF_KEYWORD,
                    FOR_KEYWORD,
                    WHILE_KEYWORD,
                    GOTO_KEYWORD,
                    LABEL_KEYWORD,
                    RETURN_KEYWORD,
                    LOOP_END_KEYWORD,
                    END_KEYWORD,
                )
            ):
                raise ValueError(
                    f"Control flow statements ({line.split()[0]}) are not fully supported for single-line execution."
                )
            else:
                # Try to evaluate as a standalone expression? Only for REPL?
                # Let's disallow for now to be consistent with file execution.
                raise ValueError("Unknown command or invalid statement")

        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            raise ValueError(f"Error on line {line_num_for_error + 1}: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error on line {line_num_for_error + 1}: {type(e).__name__}: {e}"
            ) from e

    def handle_function_call(
        self,
        line: str,
        caller_evaluator: ExpressionEvaluator,
    ) -> Any:
        """Handles function calls with the |> operator, manages scope."""
        if self.debug:
            print(f"  Handling function call statement: {line}")
        match = re.match(r"(\(.*?\)|_)\s*\|>\s*([a-zA-Z_]\w*)", line)
        if not match:
            # Check for IMPORT statement *before* raising ValueError
            import_match = re.match(
                r"IMPORT\s+([a-zA-Z_]\w*)\s+FROM\s+([a-zA-Z_]\w*)", line
            )
            if import_match:
                raise ValueError(
                    "IMPORT statement must be on its own line and cannot be part of a function call."
                )
            # Maybe it's part of a REAS? e.g. REAS result = (...) |> func
            # Let REAS handle the expression evaluation in that case.
            # If it's a standalone line, it's an error here.
            raise ValueError("Invalid function call syntax")

        args_str, func_name = match.groups()

        if func_name not in self.functions:
            raise ValueError(f"Undefined function: '{func_name}'")

        func = self.functions[func_name]
        if self.debug:
            print(f"  Calling function '{func_name}' (returns {func.return_type})")

        # --- Argument Parsing and Evaluation ---
        evaluated_args = []
        if args_str != "_" and args_str.strip() != "()":  # Allow () for no args
            args_str_content = args_str[1:-1].strip()
            if args_str_content:
                # Simple split by comma - might fail with commas inside nested calls or literals
                # A more robust parser is needed for complex arguments.
                arg_expressions = [
                    arg.strip()
                    for arg in caller_evaluator._parse_argument_list(args_str_content)
                ]

                if len(arg_expressions) != len(func.params):
                    raise ValueError(
                        f"Function '{func_name}' expects {len(func.params)} arguments, but got {len(arg_expressions)}"
                    )

                # Evaluate arguments *in the caller's scope*
                for i, arg_expr in enumerate(arg_expressions):
                    param_name, param_type = func.params[i]
                    try:
                        arg_value = caller_evaluator.evaluate(arg_expr, param_type)
                        evaluated_args.append(arg_value)
                        if self.debug:
                            print(
                                f"    Arg '{param_name}': '{arg_expr}' -> {repr(arg_value)}"
                            )
                    except (ValueError, TypeError, IndexError) as e:
                        raise ValueError(
                            f"Error evaluating argument {i+1} ('{param_name}') for function '{func_name}': {e}"
                        )
            elif len(func.params) != 0:  # () provided but function expects args
                raise ValueError(
                    f"Function '{func_name}' expects {len(func.params)} arguments, but got 0"
                )

        elif args_str == "_" and len(func.params) != 0:
            raise ValueError(
                f"Function '{func_name}' expects {len(func.params)} arguments, but got '_' (no arguments passed)"
            )
        elif args_str == "()" and len(func.params) != 0:
            raise ValueError(
                f"Function '{func_name}' expects {len(func.params)} arguments, but got 0"
            )
        # Should be covered above, but double check
        elif len(evaluated_args) != len(func.params):
            # This case might occur if args_str is empty/invalid when params expected
            raise ValueError(
                f"Argument count mismatch for function '{func_name}': expected {len(func.params)}, got {len(evaluated_args)}"
            )

        # --- Scope Setup ---
        local_vars: Dict[str, tuple[Any, str]] = {}
        # Bind evaluated arguments to parameter names in the local scope
        for i, (param_name, param_type) in enumerate(func.params):
            local_vars[param_name] = (evaluated_args[i], param_type)
        if self.debug:
            print(f"  Function '{func_name}' local scope initialized: {local_vars}")

        # --- Execute Function Body ---
        # The function body uses its own evaluator with the local scope
        return_value = self.execute_block(func.body, local_vars)

        # --- Type Check Return Value ---
        if return_value is None and func.return_type != "void":
            print(
                f"Warning: Function '{func_name}' reached end without RETURN, but expected '{func.return_type}'. Returning None."
            )
            # Convert None to the expected type if possible (e.g., 0 for int, "" for string)
            try:
                # Use a temporary evaluator for this conversion
                temp_eval = ExpressionEvaluator({}, {}, self, self.debug)
                return_value = temp_eval._convert_type(None, "void", func.return_type)
                if self.debug:
                    print(
                        f"  Converted implicit None return to: {repr(return_value)} ({func.return_type})"
                    )
            except ValueError:
                # If conversion fails, return None anyway
                pass
        elif return_value is not None:
            # Check if the actual return value matches the declared return type
            actual_return_type = _get_python_type_name(return_value)
            if (
                func.return_type != "any"
                and actual_return_type != func.return_type
                and not (func.return_type == "float" and actual_return_type == "int")
            ):
                # Try to convert to the declared type
                try:
                    # Use a temporary evaluator for this conversion
                    temp_eval = ExpressionEvaluator({}, {}, self, self.debug)
                    converted_return_value = temp_eval._convert_type(
                        return_value, actual_return_type, func.return_type
                    )
                    if self.debug:
                        print(
                            f"  Converted return value from {actual_return_type} to {func.return_type}: {repr(converted_return_value)}"
                        )
                    return_value = converted_return_value
                except ValueError as e:
                    raise ValueError(
                        f"Function '{func_name}' returned type '{actual_return_type}' but expected '{func.return_type}', and conversion failed: {e}"
                    )

        if self.debug:
            print(
                f"  Function '{func_name}' finished execution, returning: {repr(return_value)}"
            )

        # Return the final (potentially type-checked and converted) value
        return return_value

    def run(self, filepath: str):
        """Runs the interpreter on the given script file."""
        try:
            self.script_directory = os.path.dirname(os.path.abspath(filepath))
            if self.debug:
                print(
                    f"[Interpreter] Main script directory set to: {self.script_directory}"
                )
            with open(filepath, "r") as file:
                lines = file.readlines()
            # Pre-scan for labels and function definitions
            self.pre_scan(lines)
            # Execute lines
            self.execute_block(lines)
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except ValueError as e:
            # Catch errors propagated from execute_block/execute_line
            print(f"Runtime Error: {e}")
            # Optionally print line number if available/tracked
        except Exception as e:
            print(f"Unexpected Error: {type(e).__name__}: {e}")
            # Consider adding traceback for unexpected errors during debugging
            # import traceback
            # traceback.print_exc()

    def run_repl(self):
        """Run the interpreter in REPL (Read-Eval-Print Loop) mode."""
        print("Pepper Programming Language REPL")
        print("Type 'exit' to quit, 'help' for commands")
        # Consider adding 'vars' command instead of printing always
        # print("Type 'vars' to see current variables.")

        while True:
            # Show current variables for context (optional)
            # if self.variables:
            #     vars_str = ", ".join(
            #         f"{k}: {v[1]}" for k, v in self.variables.items()
            #     )
            #     print(f"\nVariables: {vars_str}")

            try:
                line = input(">>> ").strip()

                if not line:
                    continue
                if line.lower() == "exit":
                    break
                elif line.lower() == "help":
                    self._show_help()
                    continue
                elif line.lower() == "vars":
                    if self.variables:
                        print("Variables:")
                        for k, (v, t) in self.variables.items():
                            print(f"  {k}: {t} = {repr(v)}")
                    else:
                        print("No variables defined.")
                    continue
                elif line.lower() == "clear":
                    self.variables.clear()
                    # Also clear functions in REPL? Maybe not.
                    self.functions.clear()
                    self.labels.clear()
                    print("Global variables cleared.")
                    continue

                # Pre-scan and execute the single line
                # REPL doesn't easily support multi-line functions or jumps
                # We can try executing directly, but complex statements might fail
                if (
                    LABEL_KEYWORD in line
                    or "::(" in line
                    or GOTO_KEYWORD in line
                    or IF_KEYWORD in line
                    or FOR_KEYWORD in line
                    or WHILE_KEYWORD in line
                ):
                    print(
                        "Warning: Multi-line constructs (functions, labels, loops, complex ifs) have limited support in REPL."
                    )
                try:
                    # Update evaluator's variable view
                    self.expression_evaluator.variables = self.variables
                    self.expression_evaluator.functions = self.functions
                    # Execute in global scope
                    result = self.execute_line(line, 0, self.variables)
                    if (
                        result is not None
                    ):  # e.g., SHOW command prints, function call returns
                        # Don't print result automatically unless it's an expression?
                        # Let SHOW handle printing.
                        pass
                except ValueError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Unexpected Error: {type(e).__name__}: {e}")

            except EOFError:  # Handle Ctrl+D
                break
            except KeyboardInterrupt:  # Handle Ctrl+C
                print("\nKeyboardInterrupt")
                # Optionally offer to exit or clear state

        print("Exiting REPL.")

    def _show_help(self):
        """Show help information for REPL mode."""
        help_text = """
    Pepper REPL Help:

    Commands:
    LET x: type = value    - Declare a global variable (e.g., LET count:int = 0)
    REAS x = value        - Reassign a global variable (e.g., REAS count = count + 1)
    SHOW(expression)      - Evaluate expression and print the result (e.g., SHOW(count * 2))
    ? seed_value          - Set the random number generator seed (e.g., ? 12345 or ? my_seed_string)
    vars                  - Show currently defined global variables and their values.
    clear                 - Clear all global variables.
    exit                  - Exit the REPL.
    help                  - Show this help message.

    Types: int, float, string, bool, list, any, void

    Expressions:
    Literals: 10, 3.14, "hello", true, false, [1, "a", true]
    Variables: x, count
    Input: INPT("Prompt")
    Type Convert: expr :> type (e.g., "123" :> int)
    Arithmetic (RPN): 5 3 + (space-separated Reverse Polish Notation)
                        ? (evaluates to random float in RPN)
    Boolean: a && b (equal), a &$$& b (not equal), a @$@ b (and), a #$# b (or)
            a < b, a > b, a <= b, a >= b
            ~@ expr (negation)
    String Concat: "Hello " + name + "!"
    List Ops (var is list/string):
        var value [a]     - Append value
        var value [r]     - Remove first occurrence of value
        var value [n] pos - Insert value at 1-based position 'pos'
        var value [p] new - Replace first occurrence of 'value' with 'new'
        var value [P] new - Replace ALL occurrences of 'value' with 'new'
        var [l]           - Get length
        var [i] index     - Get item at 1-based 'index'
        var [f] value     - Find 1-based index of 'value' (0 if not found)
        var [?]           - Get random element

    Notes:
    - REPL primarily operates on global variables.
    - Multi-line blocks (IF, FOR, WHILE, functions) have limited support.
    - Statements generally don't produce output unless using SHOW.
    """
        print(help_text)

    def pre_scan(self, lines: List[str]):
        """Scans lines for labels and function definitions before execution."""
        self.labels.clear()
        self.functions.clear()
        line_number = 0
        while line_number < len(lines):
            line = lines[line_number].split("%%")[0].strip()
            if not line:
                line_number += 1
                continue

            let_match = re.match(r"LET\s+([a-zA-Z_]\w*)\s*:\s*(\w+)\s*=\s*(.+)", line)
            if let_match:
                var_name, var_type, expression = let_match.groups()
                try:
                    # Attempt to evaluate the expression as a literal
                    # But ONLY if it contains no variables!
                    if not re.search(r"[a-zA-Z_]\w*", expression):
                        value = self.expression_evaluator._evaluate_literal(
                            expression, None
                        )
                        if value is not None:
                            # Convert value to string representation if needed
                            # This handles bools, numbers, strings
                            replacement = repr(value)
                            # Reconstruct
                            line = f"LET {var_name}:{var_type} = {replacement}"
                            lines[line_number] = line  # Update original line
                            if self.debug:
                                print(
                                    f"  [Pre-scan] Constant folded '{expression}' to '{replacement}' on line {line_number + 1}"
                                )

                except (ValueError, SyntaxError):
                    # Not a constant expression, ignore
                    pass
            # Find Labels
            if line.startswith(LABEL_KEYWORD):
                match = re.match(r"LBL\s+([a-zA-Z_]\w*)\s*;", line)
                if match:
                    label_name = match.group(1)
                    if label_name in self.labels:
                        raise ValueError(
                            f"Duplicate label '{label_name}' defined on line {line_number + 1}"
                        )
                    # Store 1-based line number for GOTO convenience
                    self.labels[label_name] = line_number + 1
                    if self.debug:
                        print(f"Found label '{label_name}' at line {line_number + 1}")
                else:
                    # Raise error during pre-scan for invalid label syntax
                    raise ValueError(
                        f"Invalid label definition on line {line_number + 1}: {line}"
                    )
                line_number += 1
            # Find Function Definitions
            elif "::(" in line:  # Potential function start
                try:
                    lines_in_func = self.handle_function_declaration(
                        lines, line_number, store_function=True
                    )
                    # Skip the entire function body during pre-scan
                    line_number += lines_in_func + 1  # +1 to move past the return line
                except ValueError as e:
                    # Propagate errors found during function declaration parsing
                    raise ValueError(
                        f"Error in function definition near line {line_number + 1}: {e}"
                    )
            else:
                line_number += 1
        if self.debug:
            print(
                f"Pre-scan complete. Labels: {self.labels}, Functions: {list(self.functions.keys())}"
            )

    def execute_block(
        self, lines: List[str], local_vars: Optional[Dict[str, tuple[Any, str]]] = None
    ) -> Any:
        """
        Executes a block of lines (main script or function body).
        Manages execution flow (loops, ifs, goto).
        Returns the value from a RETURN statement, if encountered.
        """
        current_vars = local_vars if local_vars is not None else self.variables
        # Create a new evaluator instance for this block, sharing functions but using current_vars
        evaluator = ExpressionEvaluator(current_vars, self.functions, self, self.debug)

        line_ptr = 0  # 0-based index for line execution
        max_lines = len(lines)
        # Stores (type, start_line_ptr, <optional_loop_var>) for loops
        loop_stack = []
        conditional_stack = []  # Helps manage nested IF/ELIF/ELSE state

        while 0 <= line_ptr < max_lines:
            line = lines[line_ptr].split("%%")[0].strip()
            current_line_num = line_ptr + 1  # 1-based for messages

            # Skip empty lines, comments, labels (already processed), and function defs (already processed)
            if not line or line.startswith("%%") or line.startswith(LABEL_KEYWORD):
                line_ptr += 1
                continue
            # Also skip function definition lines identified during pre-scan (simplistic check)
            if "::(" in line and line.endswith("->"):
                # Need a robust way to skip the whole function body here
                # Let's rely on handle_function_declaration to calculate skip count
                try:
                    lines_to_skip = self.handle_function_declaration(
                        lines, line_ptr, store_function=False
                    )
                    line_ptr += lines_to_skip + 1  # Skip body and return line
                    continue
                except ValueError:
                    # Ignore errors here, pre-scan should have caught them
                    line_ptr += 1
                    continue

            if self.debug:
                print(f"\n[Exec Line {current_line_num}]> {line}")
                if current_vars != self.variables:  # If in function scope
                    print(f"  Local Vars: {current_vars}")

            try:
                if line.startswith("?"):  # Quick check before regex
                    seed_match = re.match(r"\?\s+(.+)", line)
                    if seed_match:
                        seed_value_str = seed_match.group(1).strip()
                        try:
                            # Attempt conversion to int first, as it's common for seeds
                            seed_value_for_func = int(seed_value_str)
                            random.seed(seed_value_for_func)
                            if self.debug:
                                print(
                                    f"  Seeded random generator with int: {seed_value_for_func}"
                                )
                        except ValueError:
                            # If int conversion fails, use the string itself.
                            # random.seed() accepts various hashable types.
                            seed_value_for_func = seed_value_str
                            random.seed(seed_value_for_func)
                            if self.debug:
                                print(
                                    f"  Seeded random generator with string: '{seed_value_for_func}'"
                                )

                        line_ptr += 1  # Move to next line
                        continue  # Finished processing the seed command for this line
                    else:
                        # Handle cases like just "?" or "?seed" (no space/value)
                        raise ValueError(
                            "Invalid seed syntax. Expected '? <seed_value>'"
                        )
                # --- Control Flow ---
                elif line.startswith(GOTO_KEYWORD):
                    # Condition is optional
                    match = re.match(r"GOTO\s+(\w+)\s*(?:;\s*(.+))?", line)
                    if not match:
                        raise ValueError("Invalid GOTO syntax")
                    target, condition = match.groups()

                    target_line_num = -1  # 1-based target line
                    if target.isdigit():
                        target_line_num = int(target)
                    elif target in self.labels:
                        target_line_num = self.labels[target]
                    else:
                        raise ValueError(
                            f"GOTO target '{target}' is not a valid line number or defined label"
                        )

                    if not (1 <= target_line_num <= max_lines):
                        raise ValueError(
                            f"GOTO target line number {target_line_num} is out of bounds (1-{max_lines})"
                        )

                    should_jump = True  # Default to jump if no condition
                    if condition and condition.strip():
                        condition_value = evaluator.evaluate(condition.strip(), "bool")
                        should_jump = bool(condition_value)
                        if self.debug:
                            print(
                                f"  GOTO condition '{condition.strip()}' evaluated to {should_jump}"
                            )

                    if should_jump:
                        if self.debug:
                            print(f"  Jumping to line {target_line_num}")
                        # Set next line_ptr (0-based)
                        line_ptr = target_line_num - 1
                        continue  # Start next iteration at the target line
                    else:
                        # Condition false, just proceed to the next line
                        line_ptr += 1
                        continue

                elif line.startswith(IF_KEYWORD):
                    match = re.match(r"IF\s+(.+?)\s+DO", line)
                    if not match:
                        raise ValueError("Invalid IF syntax")
                    condition = match.group(1)
                    cond_val = evaluator.evaluate(condition, "bool")
                    if self.debug:
                        print(f"  IF condition '{condition}' -> {cond_val}")
                    conditional_stack.append(
                        {"type": "if", "executed": cond_val, "line": line_ptr}
                    )
                    if not cond_val:
                        # Skip to the next ELIF, ELSE, or END for this IF level
                        line_ptr = self._find_matching_conditional_end(
                            lines, line_ptr, ["elif", "else", "end"]
                        )
                        continue  # Continue execution from the found line

                elif line.startswith(ELIF_KEYWORD):
                    if not conditional_stack or conditional_stack[-1]["type"] not in (
                        "if",
                        "elif",
                    ):
                        raise ValueError("ELIF without matching IF/ELIF")
                    match = re.match(r"ELIF\s+(.+?)\s+DO", line)
                    if not match:
                        raise ValueError("Invalid ELIF syntax")

                    # Only evaluate if no previous block in this chain executed
                    if conditional_stack[-1]["executed"]:
                        # Skip to the END of the current IF structure
                        line_ptr = self._find_matching_conditional_end(
                            lines, conditional_stack[-1]["line"], ["end"]
                        )
                        continue
                    else:
                        condition = match.group(1)
                        cond_val = evaluator.evaluate(condition, "bool")
                        if self.debug:
                            print(f"  ELIF condition '{condition}' -> {cond_val}")
                        conditional_stack[-1]["type"] = "elif"  # Update state
                        # Mark if this one executed
                        conditional_stack[-1]["executed"] = cond_val
                        if not cond_val:
                            # Skip to the next ELIF, ELSE, or END
                            line_ptr = self._find_matching_conditional_end(
                                lines, line_ptr, ["elif", "else", "end"]
                            )
                            continue

                elif line.startswith(ELSE_KEYWORD):
                    if not conditional_stack or conditional_stack[-1]["type"] not in (
                        "if",
                        "elif",
                    ):
                        raise ValueError("ELSE without matching IF/ELIF")
                    # Optional DO? Let's require it for consistency
                    match = re.match(r"ELSE\s+DO", line)
                    if not match:
                        raise ValueError("Invalid ELSE syntax, expected 'ELSE DO'")

                    if conditional_stack[-1]["executed"]:
                        # Previous block executed, skip to END
                        line_ptr = self._find_matching_conditional_end(
                            lines, conditional_stack[-1]["line"], ["end"]
                        )
                        continue
                    else:
                        # Execute this ELSE block
                        conditional_stack[-1]["type"] = "else"  # Update state
                        conditional_stack[-1]["executed"] = True

                elif line.startswith(END_KEYWORD):
                    # Optional semicolon? Let's require it.
                    match = re.match(r"END\s*", line)
                    if not match:
                        raise ValueError("Invalid END syntax, expected 'END'")
                    if not conditional_stack:
                        raise ValueError("END without matching IF")
                    conditional_stack.pop()  # Exit the current IF/ELIF/ELSE structure

                elif line.startswith(FOR_KEYWORD):
                    match = re.match(
                        r"FOR\s+([a-zA-Z_]\w*)\s+FROM\s+(.+?)\s+TO\s+(.+?)\s+DO", line
                    )
                    if not match:
                        raise ValueError("Invalid FOR syntax")
                    var_name, start_expr, end_expr = match.groups()

                    start_val = evaluator.evaluate(start_expr, "int")
                    end_val = evaluator.evaluate(end_expr, "int")

                    # Check if this is the first entry into the loop
                    if (
                        not loop_stack
                        or loop_stack[-1][0] != "for"
                        or loop_stack[-1][1] != line_ptr
                    ):
                        # Initialize loop variable and push state
                        current_vars[var_name] = (start_val, "int")
                        loop_stack.append(("for", line_ptr, var_name, end_val))
                        if self.debug:
                            print(
                                f"  FOR loop start: {var_name}={start_val}, end={end_val}"
                            )
                    else:
                        # Increment existing loop variable
                        current_val = current_vars[var_name][0]
                        current_vars[var_name] = (current_val + 1, "int")
                        if self.debug:
                            print(
                                f"  FOR loop increment: {var_name}={current_vars[var_name][0]}"
                            )

                    # Check loop condition
                    if current_vars[var_name][0] > end_val:  # Loop finished
                        if self.debug:
                            print("  FOR loop finished")
                        loop_stack.pop()
                        # Skip to end of loop body
                        line_ptr = self._find_matching_loop_end(lines, line_ptr)
                        # line_ptr will be incremented at the end, so skip LOOP_END line itself
                        # continue

                elif line.startswith(WHILE_KEYWORD):
                    match = re.match(r"WHILE\s+(.+?)\s+DO", line)
                    if not match:
                        raise ValueError("Invalid WHILE syntax")
                    condition = match.group(1)

                    # Check if first entry
                    if (
                        not loop_stack
                        or loop_stack[-1][0] != "while"
                        or loop_stack[-1][1] != line_ptr
                    ):
                        loop_stack.append(("while", line_ptr, condition))

                    # Evaluate condition
                    cond_val = evaluator.evaluate(condition, "bool")
                    if self.debug:
                        print(f"  WHILE condition '{condition}' -> {cond_val}")

                    if not cond_val:  # Loop finished or condition initially false
                        if self.debug:
                            print("  WHILE loop finished/skipped")
                        if (
                            loop_stack
                            and loop_stack[-1][0] == "while"
                            and loop_stack[-1][1] == line_ptr
                        ):
                            loop_stack.pop()  # Pop only if it was the current loop entry
                        # Skip to end of loop body
                        line_ptr = self._find_matching_loop_end(lines, line_ptr)
                        # continue

                elif line.startswith(LOOP_END_KEYWORD):
                    # Optional semicolon? Require it.
                    match = re.match(r"LOOP_END\s*", line)
                    if not match:
                        raise ValueError("Invalid LOOP_END syntax, expected 'LOOP_END'")
                    if not loop_stack:
                        raise ValueError(
                            f"{LOOP_END_KEYWORD} without matching FOR or WHILE"
                        )

                    loop_type, start_line_ptr, *_ = loop_stack[-1]

                    if loop_type == "for":
                        # Jump back to the FOR line to increment and re-check
                        line_ptr = start_line_ptr
                        continue  # Re-execute the FOR line
                    elif loop_type == "while":
                        # Jump back to the WHILE line to re-evaluate condition
                        line_ptr = start_line_ptr
                        continue  # Re-execute the WHILE line
                    else:  # Should not happen
                        raise ValueError("Internal error: Unknown loop type on stack")

                # --- Return Statement ---
                elif line.startswith(RETURN_KEYWORD):
                    if local_vars is None:  # In global scope
                        raise ValueError(
                            "RETURN statement can only be used inside a function"
                        )
                    expr = line[len(RETURN_KEYWORD) :].strip()
                    return_value = None
                    if expr:
                        # Need function's expected return type here
                        # This requires passing function context down, or looking it up.
                        # For now, evaluate without specific type.
                        return_value = evaluator.evaluate(
                            expr, None
                        )  # Evaluate in local scope
                    if self.debug:
                        print(f"  Function returning: {repr(return_value)}")
                    return return_value  # Exit the function execution

                # --- Basic Commands ---
                elif line.startswith(LET_KEYWORD):
                    self.handle_let(line, current_vars, evaluator)
                elif line.startswith(REAS_KEYWORD):
                    self.handle_reas(line, current_vars, evaluator)
                elif line.startswith(SHOW_KEYWORD):
                    self.handle_show(line, evaluator)
                elif "|>" in line:  # Function call as a statement
                    self.handle_function_call(line, evaluator)
                elif line.startswith(IMPORT_KEYWORD):
                    match = re.match(
                        r"IMPORT\s+([a-zA-Z_]\w*)\s+FROM\s+([a-zA-Z0-9_./\\-]+)", line
                    )
                    if not match:
                        raise ValueError(
                            "Invalid IMPORT syntax. Expected: IMPORT function_name FROM module_name"
                        )
                    function_name, module_name = match.groups()
                    try:
                        if self.debug:
                            print(
                                f"  Executing IMPORT: '{function_name}' FROM '{module_name}' at line {current_line_num}"
                            )
                        # Load the function using the main interpreter's method
                        imported_function = self._load_module(
                            module_name, function_name
                        )
                        # Add the loaded function to the *main* interpreter's function dictionary
                        # This makes it available globally, even if imported inside another function.
                        self.functions[function_name] = imported_function
                        if self.debug:
                            print(
                                f"  Import successful. '{function_name}' added to functions: {list(self.functions.keys())}"
                            )
                        # No need to print confirmation during script execution unless debugging

                    except (FileNotFoundError, ValueError) as e:
                        # Add line number context to the error
                        raise ValueError(
                            f"Error during import on line {current_line_num}: {e}"
                        ) from e
                    except Exception as e:
                        raise RuntimeError(
                            f"Unexpected error during import on line {current_line_num}: {type(e).__name__}: {e}"
                        ) from e

                    # Crucial: After handling IMPORT, move to the next line
                    line_ptr += 1
                    continue  # Skip the rest of the loop for this linepass
                # Add other simple statements here if needed

                # --- Unknown Statement ---
                else:
                    # Before declaring invalid, try evaluating as a standalone expression?
                    # E.g. allow `x + 1` on a line? Let's disallow for now.
                    raise ValueError("Unknown or invalid statement")

            except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                # Catch evaluation and runtime errors
                raise ValueError(f"Error on line {current_line_num}: {e}") from e
            except Exception as e:
                # Catch unexpected errors
                raise RuntimeError(
                    f"Unexpected error on line {current_line_num}: {type(e).__name__}: {e}"
                ) from e

            # --- Move to next line ---
            line_ptr += 1

        # End of block reached
        if loop_stack:
            raise ValueError(
                f"Reached end of block with unclosed loop(s) starting on line(s): {[line_num[1]+1 for line_num in loop_stack]}"
            )
        if conditional_stack:
            raise ValueError(
                f"Reached end of block with unclosed IF/ELIF/ELSE structure(s) starting on line(s): {[c['line']+1 for c in conditional_stack]}"
            )

        # If this is a function block, return None by default if no RETURN was hit
        if local_vars is not None:
            return None
        # Global block execution doesn't explicitly return a value
        return None

    def _find_matching_loop_end(self, lines, start_line_ptr):
        """Finds the line index of the matching LOOP_END for a FOR/WHILE."""
        nesting_level = 0
        ptr = start_line_ptr + 1
        while ptr < len(lines):
            line = lines[ptr].split("%%")[0].strip()
            if line.startswith((FOR_KEYWORD, WHILE_KEYWORD)):
                nesting_level += 1
            elif line.startswith(LOOP_END_KEYWORD):
                match = re.match(r"LOOP_END\s*", line)
                if match:
                    if nesting_level == 0:
                        return ptr  # Found the matching end
                    nesting_level -= 1
            ptr += 1
        raise ValueError(
            f"Missing LOOP_END for loop starting on line {start_line_ptr + 1}"
        )

    def _find_matching_conditional_end(self, lines, start_line_ptr, targets):
        """Finds the next ELIF, ELSE, or END at the same nesting level."""
        nesting_level = 0
        ptr = start_line_ptr + 1
        while ptr < len(lines):
            line = lines[ptr].split("%%")[0].strip()
            if line.startswith(IF_KEYWORD):
                nesting_level += 1
            elif line.startswith(END_KEYWORD):
                match = re.match(r"END\s*", line)
                if match:
                    if nesting_level == 0:
                        # Found the END for the starting IF
                        return (
                            ptr if "end" in targets else ptr
                        )  # Adjust based on target needs
                    nesting_level -= 1
            elif nesting_level == 0:
                # Check for targets only at the same level
                if any(line.startswith(k.upper()) for k in targets):
                    # Check syntax more strictly?
                    if line.startswith(ELIF_KEYWORD) and "elif" in targets:
                        return ptr
                    if line.startswith(ELSE_KEYWORD) and "else" in targets:
                        return ptr
                    # Add END check here too?
            ptr += 1
        # If searching for END and not found
        if "end" in targets:
            raise ValueError(
                f"Missing END for IF block starting on line {start_line_ptr + 1}"
            )
        # If searching for ELIF/ELSE and not found, return line count (effectively end of block)
        return len(lines)

    # Remove handle_for, handle_while, handle_if - logic moved into execute_block

    # def execute_line(self, line, line_num_for_error, current_vars):
    #     """
    #     Executes a single line in the given variable context.
    #     Used primarily by REPL or potentially simple sub-execution.
    #     NOTE: This is a simplified version and won't handle control flow correctly.
    #           The main execution logic is in `execute_block`.
    #     """
    #     evaluator = ExpressionEvaluator(current_vars, self.functions, self, self.debug)
    #     line = line.strip()
    #     if not line or line.startswith("%%"):
    #         return None
    #
    #     try:
    #         # --- Seed Command (NEW for REPL) ---
    #         if line.startswith("?"):  # Quick check before regex
    #             seed_match = re.match(r"\?\s+(.+)", line)
    #             if seed_match:
    #                 seed_value_str = seed_match.group(1).strip()
    #                 try:
    #                     seed_value_for_func = int(seed_value_str)
    #                     random.seed(seed_value_for_func)
    #                     if self.debug:
    #                         print(
    #                             f"  Seeded random generator with int: {seed_value_for_func}"
    #                         )
    #                 except ValueError:
    #                     seed_value_for_func = seed_value_str
    #                     random.seed(seed_value_for_func)
    #                     if self.debug:
    #                         print(
    #                             f"  Seeded random generator with string: '{seed_value_for_func}'"
    #                         )
    #                 return None  # Seed command handled, return nothing
    #             else:
    #                 raise ValueError("Invalid seed syntax. Expected '? <seed_value>'")
    #         elif line.startswith(LET_KEYWORD):
    #             return self.handle_let(line, current_vars, evaluator)
    #         elif line.startswith(REAS_KEYWORD):
    #             return self.handle_reas(line, current_vars, evaluator)
    #         elif line.startswith(SHOW_KEYWORD):
    #             return self.handle_show(line, evaluator)
    #         elif "|>" in line:  # Function call statement
    #             return self.handle_function_call(line, evaluator)
    #         elif (
    #             "::(" in line
    #         ):  # Function definition (only declaration in REPL context)
    #             print("Function defined (syntax checked).")
    #             # Assume void return for REPL def
    #             return self.handle_function_declaration(
    #                 [line, "<-void"], 0, store_function=True
    #             )
    #         elif line.startswith(
    #             (
    #                 IF_KEYWORD,
    #                 FOR_KEYWORD,
    #                 WHILE_KEYWORD,
    #                 GOTO_KEYWORD,
    #                 LABEL_KEYWORD,
    #                 RETURN_KEYWORD,
    #                 LOOP_END_KEYWORD,
    #                 END_KEYWORD,
    #             )
    #         ):
    #             raise ValueError(
    #                 f"Control flow statements ({line.split()[0]}) are not fully supported for single-line execution."
    #             )
    #         else:
    #             # Try to evaluate as a standalone expression? Only for REPL?
    #             # Let's disallow for now to be consistent with file execution.
    #             raise ValueError("Unknown command or invalid statement")
    #
    #     except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
    #         raise ValueError(f"Error on line {line_num_for_error + 1}: {e}") from e
    #     except Exception as e:
    #         raise RuntimeError(
    #             f"Unexpected error on line {line_num_for_error + 1}: {type(e).__name__}: {e}"
    #         ) from e
    #
    def handle_let(self, line, current_vars, evaluator):
        """Handles LET statements in the specified variable scope."""
        match = re.match(r"LET\s+([a-zA-Z_]\w*)\s*:\s*(\w+)\s*=\s*(.+)", line)
        if not match:
            raise ValueError("Invalid LET syntax")

        var_name, var_type, expression = match.groups()
        if self.debug:
            print(
                f"  Parsed LET: var_name={var_name}, var_type={var_type}, expression={expression}"
            )

        if var_type not in supported_types:
            raise ValueError(f"Unsupported data type: {var_type}")
        if var_name in current_vars:
            # Allow redefining in REPL maybe, but error in script?
            # Let's error consistently for now.
            raise ValueError(
                f"Variable '{var_name}' already exists. Use REAS to reassign."
            )

        # Use the provided evaluator which uses current_vars
        value = evaluator.evaluate(expression, var_type)

        # Check if the evaluated type matches the declared type after conversion attempt
        final_type = _get_python_type_name(value)
        # allow int to be assigned to float
        if (
            var_type != "any"
            and final_type != var_type
            and not (var_type == "float" and final_type == "int")
        ):
            # This check might be redundant if evaluate enforces expected_type strictly
            print(
                f"Warning: Evaluated type '{final_type}' differs from declared type '{var_type}' for '{var_name}'. Value: {repr(value)}"
            )
            # Re-convert to be sure? Or trust evaluate? Let's trust evaluate for now.

        current_vars[var_name] = (value, var_type)
        if self.debug:
            print(f"  LET Declared: {var_name} = {repr(value)} (type: {var_type})")

    def handle_reas(self, line, current_vars, evaluator):
        """Handles REAS statements in the specified variable scope."""
        # Handle list operations disguised as REAS first
        # Example: REAS myList = myList value [a]
        # Check if RHS starts with var name
        list_op_match = re.match(r"REAS\s+([a-zA-Z_]\w*)\s*=\s*(\1\s+.+)", line)
        if list_op_match:
            var_name, list_op_expr = list_op_match.groups()
            if var_name not in current_vars:
                raise ValueError(f"Variable '{var_name}' not defined.")
            # Try evaluating the RHS as a potential list operation expression
            try:
                new_value = evaluator.evaluate(
                    # Expect original type
                    list_op_expr,
                    current_vars[var_name][1],
                )
                # Evaluator's list op should have modified in-place for lists,
                # or returned new value for strings. Reassign necessary for strings.
                current_vars[var_name] = (new_value, current_vars[var_name][1])
                if self.debug:
                    print(f"  REAS (List Op): {var_name} = {repr(new_value)}")
                return  # Done with list operation reassignment
            except ValueError:
                # If it fails, it might be a regular reassignment below
                if self.debug:
                    print("  List op eval failed, trying regular REAS")
                pass

        # Regular REAS
        match = re.match(r"REAS\s+([a-zA-Z_]\w*)\s*=\s*(.+)", line)
        if not match:
            raise ValueError(
                "Invalid REAS syntax. You may be specifing a type where none is needed"
            )

        var_name, expression = match.groups()

        if var_name not in current_vars:
            raise ValueError(
                f"Variable '{var_name}' does not exist. Use LET to declare."
            )

        original_type = current_vars[var_name][1]
        if self.debug:
            print(
                f"  Parsed REAS: var_name={var_name} (type: {original_type}), expression={expression}"
            )

        # Evaluate the expression, expecting the variable's original type
        value = evaluator.evaluate(expression, original_type)

        # Type check after evaluation (optional, depends on strictness)
        final_type = _get_python_type_name(value)
        if (
            original_type != "any"
            and final_type != original_type
            and not (original_type == "float" and final_type == "int")
        ):
            print(
                f"Warning: Reassigning '{var_name}' of type '{original_type}' with value of type '{final_type}'. Value: {repr(value)}"
            )
            # Allow reassignment but warn? Or error? Let's allow with warning.

        # Keep original declared type
        current_vars[var_name] = (value, original_type)
        if self.debug:
            print(
                f"  REAS Reassigned: {var_name} = {repr(value)} (type: {original_type})"
            )

    def handle_show(self, line, evaluator):
        """Handles SHOW statements."""
        match = re.match(
            r"SHOW\((.+)\)", line, re.DOTALL
        )  # Allow multiline expressions
        if not match:
            raise ValueError("Invalid SHOW syntax, expected SHOW(...)")

        expression = match.group(1).strip()
        if self.debug:
            print(f"  Parsed SHOW: expression={expression}")
            # Show context used by evaluator
            print(f"  SHOW Vars: {evaluator.variables}")
        try:
            # Evaluate expecting *any* type, let print handle formatting
            value = evaluator.evaluate(expression, None)
            if self.debug or self.debug_show:
                # Use repr for debug clarity
                print(f"SHOW Output: {repr(value)}")
            else:
                # Smart printing: avoid None, format lists/bools nicely
                if value is None:
                    # Maybe print "None" or "(void)"? Let's print nothing.
                    pass
                elif isinstance(value, bool):
                    print(str(value).lower())  # print 'true'/'false'
                elif isinstance(value, list):
                    # Print list elements separated by space? Or Python-like repr?
                    # Let's use standard print for now.
                    print(value)
                else:
                    print(value)
        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            # Catch evaluation errors during SHOW
            print(f"Error in SHOW expression '{expression}': {e}")
            # Don't raise, just print error for SHOW

    def handle_function_declaration(
        self, lines: List[str], start_index: int, store_function: bool
    ) -> int:
        """
        Parses function declaration, optionally stores it, and returns number of lines in the function body+return.
        Raises ValueError on syntax errors.
        """
        decl_line = lines[start_index].strip()
        # Allow spaces around ::
        match = re.match(r"([a-zA-Z_]\w*)\s*::\s*\((.*?)\)\s*->", decl_line)
        if not match:
            raise ValueError(f"Invalid function declaration syntax: {decl_line}")

        func_name, params_str = match.groups()
        if self.debug and store_function:
            print(f"  Parsing function decl: {func_name}")

        # Parse parameters
        params: List[tuple[str, str]] = []
        param_names = set()
        if params_str.strip():
            for _, param_part in enumerate(
                self.expression_evaluator._parse_argument_list(params_str)
            ):
                param_part = param_part.strip()
                if not param_part:
                    raise ValueError(
                        f"Empty parameter declaration in function '{func_name}'"
                    )
                param_match = re.match(r"([a-zA-Z_]\w*)\s*:\s*(\w+)", param_part)
                if not param_match:
                    raise ValueError(
                        f"Invalid parameter syntax '{param_part}' in function '{func_name}'"
                    )
                param_name, param_type = param_match.groups()

                if param_type not in supported_types:
                    raise ValueError(
                        f"Unsupported parameter type '{param_type}' for '{param_name}' in function '{func_name}'"
                    )
                if param_name in param_names:
                    raise ValueError(
                        f"Duplicate parameter name '{param_name}' in function '{func_name}'"
                    )
                params.append((param_name, param_type))
                param_names.add(param_name)

        # Collect function body and find return type
        body_lines = []
        return_type = None
        current_index = start_index + 1
        # Count lines belonging to the function (body + return)
        lines_processed = 0
        nesting = 0  # Track nested functions if ever supported

        while current_index < len(lines):
            # Keep indentation, split comment later
            line = lines[current_index]
            line_content = line.split("%%")[0].strip()

            # Handle end of function body
            if line_content.startswith("<-"):
                if nesting == 0:
                    parsed_return_type = line_content[2:].strip()
                    if not parsed_return_type:
                        raise ValueError(
                            f"Missing return type after '<-' in function '{func_name}' on line {current_index + 1}"
                        )
                    if parsed_return_type not in supported_types:
                        raise ValueError(
                            f"Unsupported return type '{parsed_return_type}' in function '{func_name}'"
                        )
                    return_type = parsed_return_type
                    lines_processed += 1  # Count the return line
                    break  # Function definition complete
                else:
                    # Part of nested function body
                    body_lines.append(line)
                    lines_processed += 1

            # Rudimentary check for nested functions - needs proper parsing if implemented
            elif "::(" in line_content and line_content.endswith("->"):
                nesting += 1
                body_lines.append(line)
                lines_processed += 1
            # Add line to body
            else:
                body_lines.append(line)
                lines_processed += 1

            current_index += 1
        # --- End of loop ---

        if return_type is None:
            raise ValueError(
                f"Function '{func_name}' defined starting on line {start_index + 1} has no return type declaration ('<- type')"
            )

        # Store the function if requested (during pre-scan)
        if store_function:
            if func_name in self.functions:
                raise ValueError(f"Function '{func_name}' already defined.")
            self.functions[func_name] = Function(
                func_name, params, body_lines, return_type
            )
            if self.debug:
                print(
                    f"  Stored function '{func_name}' ({len(params)} params, {len(body_lines)} body lines, returns {return_type})"
                )

        return lines_processed  # Return number of lines consumed by body + return

    # def handle_function_call(
    #     self,
    #     line: str,
    #     caller_evaluator: ExpressionEvaluator,
    # ) -> Any:
    #     """Handles function calls with the |> operator, manages scope."""
    #     if self.debug:
    #         print(f"  Handling function call statement: {line}")
    #     match = re.match(r"(\(.*?\)|_)\s*\|>\s*([a-zA-Z_]\w*)", line)
    #     if not match:
    #         # Maybe it's part of a REAS? e.g. REAS result = (...) |> func
    #         # Let REAS handle the expression evaluation in that case.
    #         # If it's a standalone line, it's an error here.
    #         raise ValueError("Invalid function call syntax")
    #
    #     args_str, func_name = match.groups()
    #
    #     if func_name not in self.functions:
    #         raise ValueError(f"Undefined function: '{func_name}'")
    #
    #     func = self.functions[func_name]
    #     if self.debug:
    #         print(f"  Calling function '{func_name}' (returns {func.return_type})")
    #
    #     # --- Argument Parsing and Evaluation ---
    #     evaluated_args = []
    #     if args_str != "_" and args_str.strip() != "()":  # Allow () for no args
    #         args_str_content = args_str[1:-1].strip()
    #         if args_str_content:
    #             # Simple split by comma - might fail with commas inside nested calls or literals
    #             # A more robust parser is needed for complex arguments.
    #             arg_expressions = [
    #                 arg.strip()
    #                 for arg in self.expression_evaluator._parse_argument_list(
    #                     args_str_content
    #                 )
    #             ]
    #
    #             if len(arg_expressions) != len(func.params):
    #                 raise ValueError(
    #                     f"Function '{func_name}' expects {len(func.params)} arguments, but got {len(arg_expressions)}"
    #                 )
    #
    #             # Evaluate arguments *in the caller's scope*
    #             for i, arg_expr in enumerate(arg_expressions):
    #                 param_name, param_type = func.params[i]
    #                 try:
    #                     arg_value = caller_evaluator.evaluate(arg_expr, param_type)
    #                     evaluated_args.append(arg_value)
    #                     if self.debug:
    #                         print(
    #                             f"    Arg '{param_name}': '{arg_expr}' -> {repr(arg_value)}"
    #                         )
    #                 except (ValueError, TypeError, IndexError) as e:
    #                     raise ValueError(
    #                         f"Error evaluating argument {i+1} ('{param_name}') for function '{func_name}': {e}"
    #                     )
    #         elif len(func.params) != 0:  # () provided but function expects args
    #             raise ValueError(
    #                 f"Function '{func_name}' expects {len(func.params)} arguments, but got 0"
    #             )
    #
    #     elif args_str == "_" and len(func.params) != 0:
    #         raise ValueError(
    #             f"Function '{func_name}' expects {len(func.params)} arguments, but got '_' (no arguments passed)"
    #         )
    #     elif args_str == "()" and len(func.params) != 0:
    #         raise ValueError(
    #             f"Function '{func_name}' expects {len(func.params)} arguments, but got 0"
    #         )
    #     # Should be covered above, but double check
    #     elif len(evaluated_args) != len(func.params):
    #         # This case might occur if args_str is empty/invalid when params expected
    #         raise ValueError(
    #             f"Argument count mismatch for function '{func_name}': expected {len(func.params)}, got {len(evaluated_args)}"
    #         )
    #
    #     # --- Scope Setup ---
    #     local_vars: Dict[str, tuple[Any, str]] = {}
    #     # Bind evaluated arguments to parameter names in the local scope
    #     for i, (param_name, param_type) in enumerate(func.params):
    #         local_vars[param_name] = (evaluated_args[i], param_type)
    #     if self.debug:
    #         print(f"  Function '{func_name}' local scope initialized: {local_vars}")
    #
    #     # --- Execute Function Body ---
    #     # The function body uses its own evaluator with the local scope
    #     return_value = self.execute_block(func.body, local_vars)
    #
    #     # --- Type Check Return Value ---
    #     if return_value is None and func.return_type != "void":
    #         print(
    #             f"Warning: Function '{func_name}' reached end without RETURN, but expected '{func.return_type}'. Returning None."
    #         )
    #         # Convert None to the expected type if possible (e.g., 0 for int, "" for string)
    #         try:
    #             # Use a temporary evaluator for this conversion
    #             temp_eval = ExpressionEvaluator({}, {}, self, self.debug)
    #             return_value = temp_eval._convert_type(None, "void", func.return_type)
    #             if self.debug:
    #                 print(
    #                     f"  Converted implicit None return to: {repr(return_value)} ({func.return_type})"
    #                 )
    #         except ValueError:
    #             # If conversion fails, return None anyway
    #             pass
    #     elif return_value is not None:
    #         # Check if the actual return value matches the declared return type
    #         actual_return_type = _get_python_type_name(return_value)
    #         if (
    #             func.return_type != "any"
    #             and actual_return_type != func.return_type
    #             and not (func.return_type == "float" and actual_return_type == "int")
    #         ):
    #             # Try to convert to the declared type
    #             try:
    #                 # Use a temporary evaluator for this conversion
    #                 temp_eval = ExpressionEvaluator({}, {}, self, self.debug)
    #                 converted_return_value = temp_eval._convert_type(
    #                     return_value, actual_return_type, func.return_type
    #                 )
    #                 if self.debug:
    #                     print(
    #                         f"  Converted return value from {actual_return_type} to {func.return_type}: {repr(converted_return_value)}"
    #                     )
    #                 return_value = converted_return_value
    #             except ValueError as e:
    #                 raise ValueError(
    #                     f"Function '{func_name}' returned type '{actual_return_type}' but expected '{func.return_type}', and conversion failed: {e}"
    #                 )
    #
    #     if self.debug:
    #         print(
    #             f"  Function '{func_name}' finished execution, returning: {repr(return_value)}"
    #         )
    #
    #     # Return the final (potentially type-checked and converted) value
    #     return return_value
