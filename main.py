import argparse
import os
import re
import ast
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random
from stdlib import StandardLibrary, _get_python_type_name


def _is_int(s):
    if not isinstance(s, str):
        s = str(s)
    return re.fullmatch(r"-?\d+", s.strip())


def _is_float(s):
    if not isinstance(s, str):
        s = str(s)
    # Basic check, allows ".5", "5." etc.
    return re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)", s.strip())


def _is_numeric_string(s):
    """Checks if a string represents an int or a float."""
    return _is_int(s) or _is_float(s)


def _raise(exception):
    """Helper to raise exceptions inside lambdas."""
    raise exception


# Constants for Keywords
FOR_KEYWORD = "FOR"
WHILE_KEYWORD = "WHILE"
LET_KEYWORD = "LET"
REAS_KEYWORD = "REAS"
SHOW_KEYWORD = "SHOW"
IF_KEYWORD = "IF"
ELIF_KEYWORD = "ELIF"
ELSE_KEYWORD = "ELSE"
END_KEYWORD = "END"
LOOP_END_KEYWORD = "LOOP_END"
INPUT_KEYWORD = "INPT"
GOTO_KEYWORD = "GOTO"
LABEL_KEYWORD = "LBL"
RETURN_KEYWORD = "RETURN"
IMPORT_KEYWORD = "IMPORT"


@dataclass
class Function:
    name: str
    params: List[tuple[str, str]]  # List of (param_name, param_type)
    body: List[str]
    return_type: str


class ExpressionEvaluator:
    def __init__(self, variables, functions, interpreter, debug=False):
        # Added functions
        self.variables = variables
        self.interpreter = interpreter
        self.script_directory: Optional[str] = None

        # Keep a reference to functions for function calls within expressions
        self.functions = functions
        self.debug = debug
        # Removed self.operators as it wasn't used consistently
        self.type_conversions = {
            ("int", "float"): float,
            ("float", "int"): int,
            ("int", "bool"): bool,
            ("float", "bool"): bool,
            ("string", "list"): lambda x: list(x),  # Simpler list conversion
            ("any", "string"): str,
            ("int", "string"): str,
            ("float", "string"): str,
            ("bool", "string"): str,
            # More sensible list to string
            ("list", "string"): lambda x: "".join(map(str, x)),
            ("string", "int"): lambda x: (
                int(x)
                if x.strip().lstrip("-").isdigit()
                else (_raise(ValueError(f"Cannot convert '{x}' to int")))
            ),
            ("string", "float"): lambda x: (
                float(x)
                if _is_float(x)
                else (_raise(ValueError(f"Cannot convert '{x}' to float")))
            ),
            ("string", "bool"): lambda x: x.lower() == "true",
            ("void", "any"): lambda _: None,
            # Return None when converting to void
            ("any", "void"): lambda _: None,
            ("list", "bool"): lambda x: bool(x),  # List to bool
            ("bool", "int"): lambda x: 1 if x else 0,  # Bool to int
            ("bool", "float"): lambda x: 1.0 if x else 0.0,  # Bool to float
        }
        self.stdlib_handler = StandardLibrary(self)

    def _parse_argument_list(self, args_content_str: str) -> List[str]:
        """Parses a comma-separated argument string, respecting brackets, parens, and quotes."""
        if not args_content_str:
            return []

        elements = []
        current_element = ""
        paren_depth = 0
        bracket_depth = 0
        in_quotes = None  # Can be ' or "

        for char in args_content_str:
            if char in ('"', "'") and in_quotes is None:
                in_quotes = char
                current_element += char
            elif char == in_quotes:
                in_quotes = None
                current_element += char
            elif char == "(" and not in_quotes:
                paren_depth += 1
                current_element += char
            elif char == ")" and not in_quotes:
                paren_depth -= 1
                current_element += char
            elif char == "[" and not in_quotes:
                bracket_depth += 1
                current_element += char
            elif char == "]" and not in_quotes:
                bracket_depth -= 1
                current_element += char
            elif (
                char == ","
                and bracket_depth == 0
                and paren_depth == 0
                and not in_quotes
            ):
                elements.append(current_element.strip())
                current_element = ""
            else:
                current_element += char

        if current_element:
            elements.append(current_element.strip())

        # Filter out potentially empty strings if there were trailing commas etc.
        return [elem for elem in elements if elem]

    def evaluate(self, expression: str, expected_type: Optional[str]):
        """Main evaluation method that routes to appropriate sub-evaluators"""
        expression = expression.strip()
        if not expression:
            raise ValueError("Cannot evaluate empty expression")

        if self.debug:
            print(
                f"Evaluating expression: '{expression}' with expected type: {expected_type}"
            )

        # 1. Direct Variable Lookup
        if re.fullmatch(r"[a-zA-Z_]\w*", expression) and expression in self.variables:
            value, var_type = self.variables[expression]
            if self.debug:
                print(
                    f"  Variable '{expression}' found: value={value}, type={var_type}"
                )
            return self._convert_type(value, var_type, expected_type)

        # 2. Literals
        literal_result = self._evaluate_literal(expression, expected_type)
        if literal_result is not None:
            if self.debug:
                print(
                    f"  Evaluated as literal: {literal_result} (requested: {expected_type})"
                )
            return literal_result  # _evaluate_literal now handles conversion

        # 3. Input Expression
        input_match = re.fullmatch(r'INPT\("([^"]*)"\)', expression)
        if input_match:
            if self.debug:
                print("  Evaluating as INPUT")
            return self._evaluate_input(input_match, expected_type)

        # --- NEW: 4. Standard Library Call ---
        stdlib_match = re.fullmatch(
            r"([a-zA-Z_]\w*)\s+FROM\s+([a-zA-Z_]\w*)\s+(.*?)\s*(\(.*\)|\_)$",
            expression,
        )
        if stdlib_match:
            if self.debug:
                print("  Evaluating as Namespaced Standard Library call")
            # Pass the match and expected_type to the new handler
            return self._evaluate_namespaced_stdlib_call(stdlib_match, expected_type)
        # --- End NEW ---

        # 4. Type Conversion Expression
        type_conv_match = re.fullmatch(r"(.+?)\s*:>\s*(\w+)", expression)
        if type_conv_match:
            if self.debug:
                print("  Evaluating as type conversion")
            # Pass expected_type
            return self._evaluate_type_conversion(type_conv_match, expected_type)

        # 5. List Literals
        list_literal_match = re.fullmatch(
            r"\[.*\]", expression, re.DOTALL
        )  # Handle multiline
        if list_literal_match:
            if self.debug:
                print("  Evaluating as list literal")
            # Evaluate list literal first, then convert if needed
            raw_list = self._evaluate_list_literal(list_literal_match)
            return self._convert_type(raw_list, "list", expected_type)

        # 6. List Operations
        list_op_match = re.fullmatch(
            # Use \S+ for non-quoted values
            r'([a-zA-Z_]\w*)\s+("[^"]*"|\S+)\s+\[(a|r|n|p|P)\](?:\s+("[^"]*"|\S+))?',
            expression,
        )
        if list_op_match:
            if self.debug:
                print("  Evaluating as list operation")
            # List operations modify in-place and return the list
            modified_list = self._evaluate_list_operation(list_op_match)
            # Update the variable directly
            list_name = list_op_match.group(1)
            self.variables[list_name] = (modified_list, self.variables[list_name][1])
            return self._convert_type(
                modified_list, self.variables[list_name][1], expected_type
            )

        list_len_match = re.fullmatch(r"([a-zA-Z_]\w*)\s+\[l\]", expression)
        if list_len_match:
            if self.debug:
                print("  Evaluating as list length")
            length = self._evaluate_list_length(list_len_match)
            return self._convert_type(length, "int", expected_type)

        list_rand_match = re.fullmatch(r"([a-zA-Z_]\w*)\s+\[\?\]", expression)
        if list_rand_match:
            if self.debug:
                print("  Evaluating as list random choice")
            choice = self._evaluate_list_random(list_rand_match)
            # Determine type of choice dynamically
            choice_type = _get_python_type_name(choice)
            return self._convert_type(choice, choice_type, expected_type)

        # Note: Using 1-based indexing for users
        list_idx_match = re.fullmatch(r"([a-zA-Z_]\w*)\s+\[i\]\s+(.+)", expression)
        if list_idx_match:
            if self.debug:
                print("  Evaluating as list indexing (1-based)")
            item = self._evaluate_list_indexing(list_idx_match)
            if item is None:  # Handle index out of range returning None
                return None
            item_type = _get_python_type_name(item)
            return self._convert_type(item, item_type, expected_type)

        # Note: Using 1-based indexing for users
        list_find_match = re.fullmatch(r"([a-zA-Z_]\w*)\s+\[f\]\s+(.+)", expression)
        if list_find_match:
            if self.debug:
                print("  Evaluating as list find (1-based index, 0 if not found)")
            index = self._evaluate_list_finding(list_find_match)
            return self._convert_type(index, "int", expected_type)

        # 7. Function Call within expression (e.g. LET x:int = (5) |> addOne)
        func_call_match = re.fullmatch(
            r"(\(.*?\)|_)\s*\|>\s*([a-zA-Z_]\w*)", expression
        )
        if func_call_match:
            if self.debug:
                print("  Evaluating as function call within expression")
            # Call the interpreter's handler, passing the expression line,
            # the current variables, and this evaluator instance for arg evaluation
            try:
                # Pass the raw expression string as the 'line' argument
                return_value = self.interpreter.handle_function_call(expression, self)
                # Get the type of the actual returned value
                return_type = _get_python_type_name(return_value)
                # Convert the return value to the type expected by the outer expression
                return self._convert_type(return_value, return_type, expected_type)
            except (ValueError, TypeError, IndexError) as e:
                # Re-raise errors from function call handling appropriately
                raise ValueError(
                    f"Error during function call '{expression}': {e}"
                ) from e

        # 8. String Concatenation (Only if '+' is present and not handled by RPN)
        # Be careful not to catch simple additions like "x + 1" if expected is numeric
        if (
            "+" in expression and '"' in expression
        ):  # Basic heuristic: quotes imply potential string concat
            try:
                if self.debug:
                    print("  Attempting string concatenation")
                result = self._evaluate_string_concatenation(expression)
                return self._convert_type(result, "string", expected_type)
            except ValueError:
                if self.debug:
                    print("  String concatenation failed, continuing...")
                pass  # Fall through if not a valid string concat

        # 9. Boolean Expressions (Infix)
        bool_match_infix = re.fullmatch(
            r"(.+?)\s*(@\$@|#\$#|&&|&\$\$&|[<>]=?)\s*(.+)", expression
        )
        if bool_match_infix:
            if self.debug:
                print("  Evaluating as infix boolean expression")
            result = self._evaluate_boolean_expression_infix(bool_match_infix)
            return self._convert_type(result, "bool", expected_type)

        # 10. Boolean Negation (Prefix)
        bool_match_neg = re.fullmatch(r"~@\s+(.+)", expression)
        if bool_match_neg:
            if self.debug:
                print("  Evaluating as boolean negation")
            result = self._evaluate_boolean_negation(bool_match_neg)
            return self._convert_type(result, "bool", expected_type)

        # 11. RPN Arithmetic/Numeric (Last resort for things that look numeric)
        # Simple check: contains digits and operators, no quotes?
        if re.search(r"[\d\.\s+\-*/%?]", expression) and not re.search(
            r'"', expression
        ):
            try:
                if self.debug:
                    print("  Attempting RPN evaluation")
                # Handle ? for random number generation within RPN
                if "?" in expression:
                    parts = expression.split()
                    for i, part in enumerate(parts):
                        if "?" in part:
                            # Simple replacement, might need refinement for complex cases
                            parts[i] = str(random.random())
                    expression = " ".join(parts)

                result = self._evaluate_rpn(expression)
                # Determine result type (int or float)
                result_type = "int" if isinstance(result, int) else "float"
                return self._convert_type(result, result_type, expected_type)
            except (ValueError, ZeroDivisionError, TypeError):
                if self.debug:
                    print("  RPN evaluation failed: {e}")
                # Fall through if RPN fails
            except IndexError:
                if self.debug:
                    print("  RPN evaluation failed: Not enough operands")

        # If nothing else matches, raise Error
        raise ValueError(f"Unable to evaluate expression: '{expression}'")

    # --- NEW: Standard Library Call Handler ---

    def _evaluate_namespaced_stdlib_call(self, match, expected_type):
        """Evaluates namespaced stdlib calls like 'upper FROM string "hello" _' or 'replace FROM string msg ("old", "new")'."""
        op_name, lib_name, base_expr_str, args_part = match.groups()
        op_name = op_name.lower()
        lib_name = lib_name.lower()
        base_expr_str = base_expr_str.strip()
        args_part = args_part.strip()

        if self.debug:
            print(
                f"  Stdlib Call: Op='{op_name}', Lib='{lib_name}', BaseExpr='{base_expr_str}', ArgsPart='{args_part}'"
            )

        # --- Look up library and operation ---
        if lib_name not in self.stdlib_handler.libraries:
            raise ValueError(f"Unknown standard library namespace: '{lib_name}'")
        library = self.stdlib_handler.libraries[lib_name]
        if op_name not in library:
            raise ValueError(f"Unknown function '{op_name}' in library '{lib_name}'")
        stdlib_method = library[op_name]

        # --- Evaluate the base expression ---
        base_value = None
        base_type = "void"
        # Handle '_' explicitly for the base expression
        if base_expr_str == "_":
            if self.debug:
                print("    Base expression is '_', treating as void")
            # Keep base_value = None, base_type = "void"
        else:
            try:
                base_value = self.evaluate(base_expr_str, None)  # Evaluate base expr
                base_type = _get_python_type_name(base_value)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating base expression '{base_expr_str}' for stdlib function '{lib_name}.{op_name}': {e}"
                ) from e

        if self.debug:
            print(
                f"    Base '{base_expr_str}' evaluated to: {repr(base_value)} (type: {base_type})"
            )

        # --- Evaluate Arguments ---
        evaluated_args = []
        if args_part == "_":
            if self.debug:
                print("    No arguments provided ('_').")
            # evaluated_args remains empty []
        elif args_part.startswith("(") and args_part.endswith(")"):
            args_content_str = args_part[1:-1].strip()
            if not args_content_str:
                if self.debug:
                    print("    Empty argument list '()'.")
                # evaluated_args remains empty []
            else:
                try:
                    # Use the robust parser to split argument expressions
                    arg_expressions = self._parse_argument_list(args_content_str)
                    if self.debug:
                        print(f"    Parsed argument expressions: {arg_expressions}")

                    for i, arg_expr in enumerate(arg_expressions):
                        try:
                            arg_value = self.evaluate(
                                arg_expr, None
                            )  # Evaluate each argument expression
                            evaluated_args.append(arg_value)
                            if self.debug:
                                print(
                                    f"      Arg {i+1} '{arg_expr}' evaluated to: {repr(arg_value)} (type: {_get_python_type_name(arg_value)})"
                                )
                        except Exception as e:
                            raise ValueError(
                                f"Error evaluating argument {i+1} ('{arg_expr}') for stdlib function '{lib_name}.{op_name}': {e}"
                            ) from e
                except Exception as e:  # Catch errors during argument parsing itself
                    raise ValueError(
                        f"Error parsing argument list '{args_part}' for stdlib function '{lib_name}.{op_name}': {e}"
                    ) from e
        else:
            # Should not happen with the regex, but safety check
            raise ValueError(
                f"Invalid arguments format: '{args_part}'. Expected '(...)' or '_'."
            )

        # --- Call the specific stdlib method ---
        # Signature: method(base_value, base_type, evaluated_args_list)
        try:
            result = stdlib_method(base_value, base_type, evaluated_args)
        except Exception as e:
            # Catch errors during the execution of the stdlib function itself
            raise ValueError(
                f"Error during execution of stdlib function '{lib_name}.{op_name}': {e}"
            ) from e

        result_type = _get_python_type_name(result)
        if self.debug:
            print(
                f"    Stdlib function '{lib_name}.{op_name}' returned: {repr(result)} (type: {result_type})"
            )

        # Convert the result to the type expected by the outer context
        return self._convert_type(result, result_type, expected_type)

    # --- End REVISED ---

    def _evaluate_input(self, match, expected_type):
        """Evaluates input expressions."""
        prompt = match.group(1)
        user_input = input(prompt + " ")

        # Try to convert directly to the expected type
        if expected_type:
            try:
                # Use a temporary conversion map based on expected_type
                temp_conversions = {
                    "int": int,
                    "float": float,
                    "bool": lambda x: x.lower() == "true",
                    "string": str,
                    "list": lambda x: [e.strip() for e in self._parse_argument_list(x)],
                }
                if expected_type in temp_conversions:
                    return temp_conversions[expected_type](user_input)
                else:  # Fallback for unknown types or None
                    return user_input
            except ValueError:
                raise ValueError(
                    f"Input '{user_input}' cannot be converted to expected type {expected_type}"
                )
        else:
            # If no expected type, return as string
            return user_input

    def _evaluate_literal(self, expression, expected_type):
        """Evaluates literal values (string, bool, int, float) with automatic type conversion,
        using stricter matching for strings."""

        # --- String Literal Check (Stricter) ---
        # Use regex to ensure ONLY a quoted string (+ optional surrounding whitespace).
        # This pattern handles escaped quotes (\") inside the string.
        lst_match = re.fullmatch(r"\s*\[(.*)\]\s*", expression)
        if lst_match:
            return None

        str_match = re.fullmatch(r'\s*"((?:[^"\\]|\\.)*)"\s*', expression)
        if str_match:
            raw_content = str_match.group(1)  # The content inside the quotes
            try:
                # Use ast.literal_eval to safely evaluate the string content,
                # correctly handling standard Python escape sequences (\n, \", \t, etc.).
                # We reconstruct the quoted string for literal_eval.
                evaluated_value = ast.literal_eval(
                    '"' + raw_content.replace('"', '\\"') + '"'
                )
            except Exception as e:
                # Fallback if literal_eval fails (highly unlikely for valid string literals)
                if self.debug:
                    print(
                        f"  Warning: ast.literal_eval failed for string content '{raw_content}': {e}. Using raw content."
                    )
                evaluated_value = raw_content  # Use the raw content as a fallback

            if self.debug:
                print(f"  Literal evaluated as string: {repr(evaluated_value)}")
            return self._convert_type(evaluated_value, "string", expected_type)

        # --- Boolean Literals ---
        if expression.lower() == "true":
            if self.debug:
                print("  Literal evaluated as bool: True")
            return self._convert_type(True, "bool", expected_type)
        if expression.lower() == "false":
            if self.debug:
                print("  Literal evaluated as bool: False")
            return self._convert_type(False, "bool", expected_type)

        # --- Numeric Literals (int or float) ---
        # Use the existing helper functions _is_int, _is_float
        if _is_int(expression):
            value = int(expression)
            if self.debug:
                print(f"  Literal evaluated as int: {value}")
            return self._convert_type(value, "int", expected_type)
        if _is_float(expression):
            value = float(expression)
            if self.debug:
                print(f"  Literal evaluated as float: {value}")
            return self._convert_type(value, "float", expected_type)

        # --- No Literal Match ---
        # Return None to indicate this expression is not a simple literal.
        return None

    def _convert_type(self, value, source_type, target_type) -> Any:
        """Enhanced type conversion system"""
        if source_type == target_type or target_type is None or target_type == "any":
            return value
        if source_type == "any":  # If source is 'any', try to guess Python type
            source_type = _get_python_type_name(value)

        key = (source_type, target_type)
        if key in self.type_conversions:
            try:
                if self.debug:
                    print(f"  Converting '{value}' ({source_type}) -> {target_type}")
                converted = self.type_conversions[key](value)
                if self.debug:
                    print(f"    -> Result: '{converted}' ({target_type})")
                return converted
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Cannot convert '{value}' from {source_type} to {target_type}: {e}"
                )
        # Allow conversion between int/float implicitly if no specific rule exists
        elif source_type in ("int", "float") and target_type in ("int", "float"):
            return float(value) if target_type == "float" else int(value)

        # Allow conversion to bool - generally True unless 0, empty, or False
        elif target_type == "bool":
            return bool(value)

        # Allow conversion to string
        elif target_type == "string":
            return str(value)

        raise ValueError(
            f"Unsupported type conversion from {source_type} to {target_type}"
        )

    def _evaluate_string_concatenation(self, expression):
        """
        Evaluates string concatenation expressions (+ operator),
        respecting variables, literals, quotes, and basic nesting.
        """
        # --- Debug Start ---
        if self.debug:
            print(f"  String concat executing on: '{expression}'")
            # Show the specific variables context this evaluator instance is using
            print(
                f"  String concat evaluator vars ID: {id(self.variables)}, Content: {self.variables}"
            )
        # --- Debug End ---

        result = ""
        parts = []
        idx = 0
        start = 0
        in_quotes = None  # Track the type of quote (' or ") currently active
        nesting_depth = 0  # Track depth of brackets '[]' and parentheses '()'

        # --- Robust Splitting Logic ---
        while idx < len(expression):
            char = expression[idx]

            # Handle entering/exiting quotes
            if char in ('"', "'"):
                if in_quotes is None:
                    in_quotes = char  # Enter quote block
                elif char == in_quotes:
                    in_quotes = None  # Exit quote block

            # Track nesting level (only outside quotes)
            elif char in ("[", "(") and in_quotes is None:
                nesting_depth += 1
            elif char in ("]", ")") and in_quotes is None:
                if nesting_depth > 0:  # Avoid going negative on mismatched brackets
                    nesting_depth -= 1

            # Split on '+' only if outside quotes and at the top level (nesting_depth == 0)
            elif char == "+" and in_quotes is None and nesting_depth == 0:
                parts.append(expression[start:idx].strip())
                start = idx + 1  # Start next part after the '+'

            idx += 1
        # --- End of Splitting Loop ---

        # Add the final part of the expression (after the last '+' or if no '+')
        parts.append(expression[start:].strip())

        # --- Debug Parts ---
        if self.debug:
            print(f"  String concat parts identified: {parts}")
        # --- Debug End ---

        # Handle case where splitting resulted in empty list (e.g., expression was just "+")
        if not parts:
            # Depending on desired behavior, could return "" or raise error.
            # Returning "" seems reasonable for an empty/invalid concat expression.
            return ""

        # --- Evaluate Each Part ---
        for part in parts:
            part = part.strip()  # Ensure part itself has no leading/trailing whitespace
            if not part:
                # Skip empty parts that might result from splitting (e.g., "a + + b")
                continue

            # --- Debug Part Evaluation ---
            if self.debug:
                print(f"      String concat evaluating part: '{part}'")
            # --- Debug End ---

            try:
                # Evaluate the part recursively using the *same* evaluator instance,
                # ensuring it uses the correct variable scope (self.variables).
                # Evaluate expecting *any* type initially.
                value = self.evaluate(part, None)

                # --- Debug Part Result ---
                if self.debug:
                    print(
                        f"      Part '{part}' evaluated to: {repr(value)} (type: {_get_python_type_name(value)})"
                    )
                # --- Debug End ---

                # Append the string representation of the evaluated value
                result += str(value)

            except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                # Improve error context if a part fails to evaluate
                raise ValueError(
                    f"Invalid part ('{part}') within string concatenation expression '{expression}': {e}"
                ) from e
            except Exception as e:
                # Catch unexpected errors during part evaluation
                raise RuntimeError(
                    f"Unexpected error evaluating part ('{part}') in string concatenation '{expression}': {type(e).__name__}: {e}"
                ) from e
        # --- End of Part Evaluation Loop ---

        # --- Debug Final Result ---
        if self.debug:
            print(f"  String concat final result: '{result}'")
        # --- Debug End ---

        return result

    def _is_list_or_string(self, name):
        """Checks if a variable exists and is a list or string."""
        if name not in self.variables:
            return False, f"Variable '{name}' not defined."
        _, var_type = self.variables[name]
        if var_type not in ("list", "string"):
            return False, f"Variable '{name}' is type '{var_type}', not list or string."
        return True, None

    def _evaluate_list_operation(self, match):
        """Evaluates list/string append/remove/insert/replace operations. Modifies in-place."""
        list_name, value_expr, operation, extra_expr = match.groups()

        is_valid, error_msg = self._is_list_or_string(list_name)
        if not is_valid:
            raise ValueError(error_msg)

        # Get the current value and type
        current_collection, collection_type = self.variables[list_name]
        # Make a copy if it's a list to avoid modifying original during evaluation if needed later
        # Although, the intention seems to be modification in place. Be careful.
        # For strings, Python strings are immutable, so assignment is needed anyway.

        # Evaluate the value expression
        # Evaluate value with no specific type expected initially
        value = self.evaluate(value_expr, None)

        # Evaluate extra expression (position or replacement value) if present
        extra_value = None
        if extra_expr:
            extra_value = self.evaluate(extra_expr, None)

        if collection_type == "list":
            # Operate on a copy then reassign if needed, or modify list directly?
            # Let's modify directly for now.
            target_list = current_collection  # Operate directly on the list
            if operation == "a":  # Append
                target_list.append(value)
            elif operation == "r":  # Remove (first occurrence)
                try:
                    target_list.remove(value)
                except ValueError:
                    print(
                        f"Warning: Value '{value}' not found in list '{list_name}' for removal."
                    )
                    # Don't raise error if not found? Or should we? Let's warn for now.
                    pass
            elif operation == "n":  # Insert
                if extra_value is None:
                    raise ValueError(
                        "Insert operation [n] requires a position argument."
                    )
                try:
                    # Note: Using 1-based indexing from user perspective
                    position = self._convert_type(
                        extra_value, _get_python_type_name(extra_value), "int"
                    )
                    if not (1 <= position <= len(target_list) + 1):
                        raise ValueError(
                            f"Insert position {position} out of range for list '{list_name}' (size {len(target_list)}). Use 1 to {len(target_list)+1}."
                        )
                    # Convert to 0-based for Python
                    target_list.insert(position - 1, value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid position for insert operation: {e}")
            elif operation == "p" or operation == "P":  # Replace/Replace All
                if extra_value is None:
                    raise ValueError(
                        "Replace operation [p/P] requires a new value argument."
                    )
                new_value = extra_value  # Already evaluated
                found = False
                for i in range(len(target_list)):
                    # Compare elements carefully (e.g., string vs number)
                    # Simple string comparison for now
                    if str(target_list[i]) == str(value):
                        target_list[i] = new_value
                        found = True
                        if operation == "p":  # Replace only first
                            break
                if not found:
                    print(
                        f"Warning: Value '{value}' not found in list '{list_name}' for replacement."
                    )

            # Return the modified list (though it was modified in-place)
            return target_list

        elif collection_type == "string":
            target_string = current_collection
            # Convert value and extra_value to string for string operations
            value_str = str(value)
            extra_value_str = str(extra_value) if extra_value is not None else None

            if operation == "a":  # Append
                target_string += value_str
            elif operation == "r":  # Remove (all occurrences)
                target_string = target_string.replace(value_str, "")
            elif operation == "n":  # Insert
                if extra_value is None:
                    raise ValueError(
                        "Insert operation [n] requires a position argument."
                    )
                try:
                    # Note: Using 1-based indexing from user perspective
                    position = self._convert_type(
                        extra_value, _get_python_type_name(extra_value), "int"
                    )
                    if not (1 <= position <= len(target_string) + 1):
                        raise ValueError(
                            f"Insert position {position} out of range for string '{list_name}' (length {len(target_string)}). Use 1 to {len(target_string)+1}."
                        )
                    # 0-based slice
                    target_string = (
                        target_string[: position - 1]
                        + value_str
                        + target_string[position - 1 :]
                    )
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid position for insert operation: {e}")
            elif operation == "p":  # Replace first
                if extra_value_str is None:
                    raise ValueError(
                        "Replace operation [p] requires a new value argument."
                    )
                target_string = target_string.replace(value_str, extra_value_str, 1)
            elif operation == "P":  # Replace all
                if extra_value_str is None:
                    raise ValueError(
                        "Replace operation [P] requires a new value argument."
                    )
                target_string = target_string.replace(value_str, extra_value_str)

            # Return the new string (strings are immutable)
            return target_string
        else:
            # Should not happen due to initial check, but added for safety
            raise ValueError(f"'{list_name}' is not a list or string.")

    def _evaluate_list_length(self, match):
        list_name = match.group(1)
        is_valid, error_msg = self._is_list_or_string(list_name)
        if not is_valid:
            raise ValueError(error_msg)
        length = len(self.variables[list_name][0])
        if self.debug:
            print(f"  List/String length result: {length}")
        return length

    def _evaluate_list_random(self, match):
        list_name = match.group(1)
        is_valid, error_msg = self._is_list_or_string(list_name)
        if not is_valid:
            raise ValueError(error_msg)
        collection = self.variables[list_name][0]
        if not collection:
            raise ValueError(
                f"Cannot get random element from empty list/string '{list_name}'"
            )
        choice = random.choice(collection)
        if self.debug:
            print(f"  List/String random choice: {choice}")
        return choice

    def _evaluate_list_indexing(self, match):
        """Evaluates list/string indexing operations. Uses 1-based indexing for user."""
        list_name, index_expression = match.groups()
        is_valid, error_msg = self._is_list_or_string(list_name)
        if not is_valid:
            raise ValueError(error_msg)

        # Evaluate the index expression, expecting an integer
        index_val = self.evaluate(index_expression.strip(), "int")
        if not isinstance(index_val, int):
            # This should ideally be caught by evaluate requesting "int", but double-check
            raise ValueError(
                f"Index must evaluate to an integer. Got: {type(index_val).__name__}"
            )

        collection = self.variables[list_name][0]
        # Convert 1-based user index to 0-based Python index
        zero_based_index = index_val - 1

        try:
            item = collection[zero_based_index]
            if self.debug:
                print(f"  List/String index [{index_val}] -> '{item}'")
            return item
        except IndexError:
            # Return None for out-of-bounds access, as per original code's warning
            print(
                f"Warning: Index {index_val} out of range for {self.variables[list_name][1]} '{list_name}' (size {len(collection)})"
            )
            return None

    def _evaluate_list_finding(self, match):
        """Evaluates list/string finding operations. Returns 1-based index or 0 if not found."""
        list_name, value_expression = match.groups()
        is_valid, error_msg = self._is_list_or_string(list_name)
        if not is_valid:
            raise ValueError(error_msg)

        # Evaluate the value to find
        value_to_find = self.evaluate(value_expression.strip(), None)

        if self.debug:
            print(
                f"  Finding value '{value_to_find}' in {self.variables[list_name][1]} '{list_name}'"
            )

        collection = self.variables[list_name][0]

        try:
            # Python's index() method works for both lists and strings
            # We need to handle potential type differences during search
            # Simple approach: convert items to string for comparison
            found_index = -1
            str_value_to_find = str(value_to_find)
            for i, item in enumerate(collection):
                if str(item) == str_value_to_find:
                    found_index = i
                    break

            if found_index != -1:
                result_index = found_index + 1  # Convert to 1-based index for user
                if self.debug:
                    print(f"    Found at 1-based index: {result_index}")
                return result_index
            else:
                if self.debug:
                    print(f"    Value '{value_to_find}' not found.")
                return 0  # Return 0 if not found

        except ValueError:
            # Should not happen with the loop above, but keep as fallback
            if self.debug:
                print(f"    Value '{value_to_find}' not found (error).")
            return 0

    def _evaluate_type_conversion(self, match, expected_type):  # Added expected_type
        """Evaluates explicit type conversion expressions like 'expr :> type'."""
        expr_to_convert, target_type = match.groups()

        if (
            target_type not in Interpreter.supported_types
        ):  # Access supported types statically
            raise ValueError(f"Unsupported target type for conversion: {target_type}")

        # First, evaluate the inner expression without a target type
        source_value = self.evaluate(expr_to_convert, None)
        source_type = _get_python_type_name(
            source_value
        )  # Get type from evaluated value

        # Now, perform the explicit conversion
        converted_value = self._convert_type(source_value, source_type, target_type)

        # Finally, convert to the overall expected type if one was provided
        return self._convert_type(converted_value, target_type, expected_type)

    def _evaluate_rpn(self, expression):
        """Evaluates RPN arithmetic expressions."""
        if self.debug:
            print(f"  Evaluating RPN: '{expression}'")
        stack = []
        tokens = expression.split()

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            # Use literal evaluation for numbers
            literal_val = self._evaluate_literal(
                token, None
            )  # Try to parse as literal first
            if isinstance(literal_val, (int, float)):
                stack.append(literal_val)
                if self.debug:
                    print(f"    RPN Push (literal): {literal_val}")
            elif token in self.variables:
                var_value, var_type = self.variables[token]
                if var_type in ("int", "float"):
                    stack.append(var_value)
                    if self.debug:
                        print(f"    RPN Push (variable '{token}'): {var_value}")
                else:
                    raise TypeError(  # Use TypeError for RPN type issues
                        f"Variable '{token}' is not numeric (type: {var_type}) in RPN expression"
                    )
            # Added ** and !/ from original code
            elif token in ("+", "-", "*", "/", "%", "**", "!/"):
                if len(stack) < 2:
                    raise IndexError(  # Use IndexError for stack underflow
                        f"Insufficient operands for RPN operator '{token}'"
                    )
                operand2 = stack.pop()
                operand1 = stack.pop()
                if self.debug:
                    print(f"    RPN Op: {operand1} {token} {operand2}")

                # Perform operation
                if token == "+":
                    result = operand1 + operand2
                elif token == "-":
                    result = operand1 - operand2
                elif token == "*":
                    result = operand1 * operand2
                elif token == "**":
                    result = operand1**operand2
                elif token == "/":
                    if operand2 == 0:
                        raise ZeroDivisionError("RPN Division by zero")
                    result = operand1 / operand2  # Keep float for division
                elif token == "%":
                    if operand2 == 0:
                        raise ZeroDivisionError("RPN Modulo by zero")
                    result = operand1 % operand2
                elif token == "!/":
                    if operand2 == 0:
                        raise ZeroDivisionError("RPN Integer division by zero")
                    result = operand1 // operand2
                else:
                    # Should not happen
                    raise ValueError(f"RPN: Unknown operator '{token}'")

                stack.append(result)
                if self.debug:
                    print(f"      = {result}")

            else:
                raise ValueError(f"Invalid RPN token: '{token}'")

        if len(stack) != 1:
            raise ValueError(f"Invalid RPN expression, final stack: {stack}")

        final_result = stack.pop()
        if self.debug:
            print(f"  RPN Result: {final_result}")
        return final_result

    def _evaluate_boolean_negation(self, match):
        """Handles prefix boolean negation '~@ expr'"""
        sub_expr = match.group(1).strip()
        # Evaluate the sub-expression, expecting a boolean
        value = self.evaluate(sub_expr, "bool")
        return not value

    def _evaluate_boolean_expression_infix(self, match):
        """Evaluates infix boolean expressions (e.g., a && b, x < 5)."""
        left_expression, operator, right_expression = match.groups()

        # Evaluate operands *without* assuming a type initially
        left_val = self.evaluate(left_expression.strip(), None)
        right_val = self.evaluate(right_expression.strip(), None)

        left_type = _get_python_type_name(left_val)
        right_type = _get_python_type_name(right_val)

        if self.debug:
            print(
                f"  Boolean Infix: '{left_val}' ({left_type}) {operator} '{right_val}' ({right_type})"
            )

        # Logical operators (AND, OR) - evaluate operands as boolean
        if operator == "@$@":  # Logical AND
            return bool(left_val) and bool(right_val)
        elif operator == "#$#":  # Logical OR
            return bool(left_val) or bool(right_val)

        # Comparison operators
        # Try numeric comparison first if both look numeric or are numeric
        can_compare_numeric = (
            left_type in ("int", "float") or _is_numeric_string(left_expression.strip())
        ) and (
            right_type in ("int", "float")
            or _is_numeric_string(right_expression.strip())
        )

        if can_compare_numeric:
            try:
                # Convert to float for comparison to handle int/float mix
                num_left = self._convert_type(left_val, left_type, "float")
                num_right = self._convert_type(right_val, right_type, "float")

                if operator == "&&":
                    return num_left == num_right
                elif operator == "&$$&":
                    return num_left != num_right
                elif operator == "<":
                    return num_left < num_right
                elif operator == ">":
                    return num_left > num_right
                elif operator == "<=":
                    return num_left <= num_right
                elif operator == ">=":
                    return num_left >= num_right
                # If we got here with a comparison operator, something's wrong
                # Fall through to string comparison? Or raise error? Let's fall through for now.
            except ValueError:
                # If conversion to float fails, fall through to string comparison
                pass

        # If not comparable as numbers, compare as strings
        str_left = str(left_val)
        str_right = str(right_val)

        if operator == "&&":
            return str_left == str_right
        elif operator == "&$$&":
            return str_left != str_right
        # For strings, <, >, <=, >= perform lexicographical comparison
        elif operator == "<":
            return str_left < str_right
        elif operator == ">":
            return str_left > str_right
        elif operator == "<=":
            return str_left <= str_right
        elif operator == ">=":
            return str_left >= str_right
        else:
            raise ValueError(f"Unsupported boolean operator: {operator}")

    def _evaluate_list_literal(self, match):
        """Evaluates list literal expressions like [1, 2, "hello", x] using ast.literal_eval"""
        list_str = match.group(0)

        if self.debug:
            print(f"  Evaluating list literal: {list_str}")

        # Handle empty list explicitly
        if list_str == "[]":
            return []

        # Need to replace variables within the list string before using literal_eval
        # This is complex and potentially unsafe if not done carefully.
        # A simple substitution might work for basic cases, but fails with nested structures or strings containing variable names.

        # --- Alternative: Manual Parsing (Adapted from original, improved) ---
        content = list_str[1:-1].strip()
        elements = []
        current_element = ""
        bracket_depth = 0
        in_quotes = None  # Can be ' or "

        for char in content:
            if char in ('"', "'") and in_quotes is None:
                in_quotes = char
                current_element += char
            elif char == in_quotes:
                in_quotes = None
                current_element += char
            elif char == "[" and not in_quotes:
                bracket_depth += 1
                current_element += char
            elif char == "]" and not in_quotes:
                bracket_depth -= 1
                current_element += char
            elif char == "," and bracket_depth == 0 and not in_quotes:
                elements.append(current_element.strip())
                current_element = ""
            else:
                current_element += char

        if current_element:
            elements.append(current_element.strip())

        # Evaluate each element recursively
        result_list = []
        for elem_expr in elements:
            if not elem_expr:
                continue
            try:
                value = self.evaluate(elem_expr, None)  # Evaluate element
                result_list.append(value)
            except ValueError as e:
                raise ValueError(f"Invalid list element expression: '{elem_expr}'. {e}")

        if self.debug:
            print(f"  Evaluated list literal result: {result_list}")
        return result_list

        # --- AST Literal Eval (Safer but less flexible) ---
        # try:
        #     # ast.literal_eval is safer than eval() but only handles literals
        #     # It CANNOT handle variables directly. We would need preprocessing.
        #     # For now, let's stick to the manual parsing which allows variables.
        #     # evaluated_list = ast.literal_eval(list_str)
        #     # if isinstance(evaluated_list, list):
        #     #     return evaluated_list
        #     # else:
        #     #     raise ValueError("Expression evaluated to non-list type via literal_eval")
        #     pass # Placeholder if using ast
        # except (ValueError, SyntaxError, TypeError) as e:
        #      raise ValueError(f"Invalid list literal syntax: {list_str}. Error: {e}")


# --- Helper functions ---


# ------------------------------------------------------------

# Part 4: `Interpreter` Class (with fixes)


class Interpreter:
    # Class attribute for supported types
    supported_types = [
        "int",
        "float",
        "string",
        "list",
        "bool",
        "any",
        "void",
    ]  # Added 'any'

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

        if var_type not in self.supported_types:
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

                if param_type not in self.supported_types:
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
                    if parsed_return_type not in self.supported_types:
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


# Part 5: `main` function and entry point (No major changes needed)


def main():
    parser = argparse.ArgumentParser(
        description="Pepper Programming Language Interpreter."
    )
    parser.add_argument(
        "filepath", nargs="?", help="The path to the script file to execute."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable detailed execution tracing."
    )
    parser.add_argument(
        "-s",
        "--debug_show",
        action="store_true",
        help="Enable debug output specifically for SHOW statements.",
    )
    args = parser.parse_args()

    interpreter = Interpreter(args.debug, args.debug_show)

    if args.filepath:
        interpreter.run(args.filepath)
    else:
        interpreter.run_repl()


if __name__ == "__main__":
    main()
