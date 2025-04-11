import ast
from typing import List, Any, Optional
import random
from stdlib import StandardLibrary, _get_python_type_name
from consts import supported_types, matchEXPR


def _is_int(s):
    if not isinstance(s, str):
        s = str(s)
    return matchEXPR(s.strip(), "int")


def _is_float(s):
    if not isinstance(s, str):
        s = str(s)
    # Basic check, allows ".5", "5." etc.
    return matchEXPR(s.strip(), "float") or matchEXPR(s.strip(), "numeric")


def _is_numeric_string(s):
    """Checks if a string represents an int or a float."""
    return _is_int(s) or _is_float(s)


def _raise(exception):
    """Helper to raise exceptions inside lambdas."""
    raise exception


class ExpressionEvaluator:
    def __init__(self, variables, functions, interpreter, debug=False):
        self.variables = variables
        self.interpreter = interpreter
        self.script_directory: Optional[str] = None

        # Keep a reference to functions for function calls within expressions
        self.functions = functions
        self.debug = debug
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

    def evaluate(self, expression: str, expected_type: Optional[str]):
        """Main evaluation method that routes to appropriate sub-evaluators"""
        expression = expression.strip()
        if not expression:
            raise ValueError("Cannot evaluate empty expression")
        self._eval_cache = {}

        if self.debug:
            print(
                f"Evaluating expression: '{expression}' with expected type: {expected_type}"
            )

        # 1. Direct Variable Lookup
        if matchEXPR(expression, "var_name") and expression in self.variables:
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
        input_match = matchEXPR(expression, "input")
        if input_match:
            if self.debug:
                print("  Evaluating as INPUT")
            return self._evaluate_input(input_match, expected_type)

        stdlib_match = matchEXPR(expression, "stdlib_call")
        if stdlib_match:
            if self.debug:
                print("  Evaluating as Namespaced Standard Library call")
            # Pass the match and expected_type to the new handler
            return self._evaluate_namespaced_stdlib_call(stdlib_match, expected_type)

        # 4. Type Conversion Expression
        type_conv_match = matchEXPR(expression, "type_conversion")
        if type_conv_match:
            if self.debug:
                print("  Evaluating as type conversion")
            # Pass expected_type
            return self._evaluate_type_conversion(type_conv_match, expected_type)

        # 5. List Literals
        list_literal_match = matchEXPR(expression, "list_literal")

        # Handle multiline
        if list_literal_match:
            if self.debug:
                print("  Evaluating as list literal")
            # Evaluate list literal first, then convert if needed
            raw_list = self._evaluate_list_literal(list_literal_match)
            return self._convert_type(raw_list, "list", expected_type)

        # 6. List Operations
        list_op_match = matchEXPR(expression, "list_operation")
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

        list_len_match = matchEXPR(expression, "list_length")
        if list_len_match:
            if self.debug:
                print("  Evaluating as list length")
            length = self._evaluate_list_length(list_len_match)
            return self._convert_type(length, "int", expected_type)

        list_rand_match = matchEXPR(expression, "list_random")
        if list_rand_match:
            if self.debug:
                print("  Evaluating as list random choice")
            choice = self._evaluate_list_random(list_rand_match)
            # Determine type of choice dynamically
            choice_type = _get_python_type_name(choice)
            return self._convert_type(choice, choice_type, expected_type)

        # Note: Using 1-based indexing for users
        list_idx_match = matchEXPR(expression, "list_index")
        if list_idx_match:
            if self.debug:
                print("  Evaluating as list indexing (1-based)")
            item = self._evaluate_list_indexing(list_idx_match)
            if item is None:  # Handle index out of range returning None
                return None
            item_type = _get_python_type_name(item)
            return self._convert_type(item, item_type, expected_type)

        # Note: Using 1-based indexing for users
        list_find_match = matchEXPR(expression, "list_find")
        if list_find_match:
            if self.debug:
                print("  Evaluating as list find (1-based index, 0 if not found)")
            index = self._evaluate_list_finding(list_find_match)
            return self._convert_type(index, "int", expected_type)

        # 7. Function Call within expression (e.g. LET x:int = (5) |> addOne)
        func_call_match = matchEXPR(expression, "function_pipe")
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
        bool_match_infix = matchEXPR(expression, "bool_infix")
        if bool_match_infix:
            if self.debug:
                print("  Evaluating as infix boolean expression")
            result = self._evaluate_boolean_expression_infix(bool_match_infix)
            return self._convert_type(result, "bool", expected_type)

        # 10. Boolean Negation (Prefix)
        bool_match_neg = matchEXPR(expression, "bool_negation")
        if bool_match_neg:
            if self.debug:
                print("  Evaluating as boolean negation")
            result = self._evaluate_boolean_negation(bool_match_neg)
            return self._convert_type(result, "bool", expected_type)

        # 11. RPN Arithmetic/Numeric (Last resort for things that look numeric)
        # Simple check: contains digits and operators, no quotes?
        if matchEXPR(expression, "maybe_rpn", True) and not matchEXPR(
            expression, "has_quotes", True
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

    def _evaluate_cached(self, expression: str, expected_type: Optional[str]):
        cache_key = (expression, expected_type)
        if cache_key in self._eval_cache:
            if self.debug:
                print(f"  Cache hit for: {expression} (type: {expected_type})")
            return self._eval_cache[cache_key]

        result = self.evaluate(expression, expected_type)
        self._eval_cache[cache_key] = result
        return result

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

    def _is_list_or_string(self, name):
        """Checks if a variable exists and is a list or string."""
        if name not in self.variables:
            return False, f"Variable '{name}' not defined."
        _, var_type = self.variables[name]
        if var_type not in ("list", "string"):
            return False, f"Variable '{name}' is type '{var_type}', not list or string."
        return True, None

    def _evaluate_literal(self, expression, expected_type):
        """Evaluates literal values (string, bool, int, float) with automatic type conversion,
        using stricter matching for strings."""

        # This pattern handles escaped quotes (\") inside the string.
        lst_match = matchEXPR(expression, "trivial_list_literal")
        if lst_match:
            return None

        str_match = matchEXPR(expression, "string_literal")
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

        if expression.lower() == "true":
            if self.debug:
                print("  Literal evaluated as bool: True")
            return self._convert_type(True, "bool", expected_type)
        if expression.lower() == "false":
            if self.debug:
                print("  Literal evaluated as bool: False")
            return self._convert_type(False, "bool", expected_type)

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

        return None

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

        if lib_name not in self.stdlib_handler.libraries:
            raise ValueError(f"Unknown standard library namespace: '{lib_name}'")
        library = self.stdlib_handler.libraries[lib_name]
        if op_name not in library:
            raise ValueError(f"Unknown function '{op_name}' in library '{lib_name}'")
        stdlib_method = library[op_name]

        base_value = None
        base_type = "void"
        # Handle '_' explicitly for the base expression
        if base_expr_str == "_":
            if self.debug:
                print("    Base expression is '_', treating as void")
            # Keep base_value = None, base_type = "void"
        else:
            try:
                base_value = self._evaluate_cached(
                    base_expr_str, None
                )  # Evaluate base expr
                base_type = _get_python_type_name(base_value)
            except Exception as e:
                raise ValueError(
                    f"Error evaluating base expression '{base_expr_str}' for stdlib function '{lib_name}.{op_name}': {e}"
                ) from e

        if self.debug:
            print(
                f"    Base '{base_expr_str}' evaluated to: {repr(base_value)} (type: {base_type})"
            )

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
                            arg_value = self._evaluate_cached(
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

    def _evaluate_type_conversion(self, match, expected_type):  # Added expected_type
        """Evaluates explicit type conversion expressions like 'expr :> type'."""
        expr_to_convert, target_type = match.groups()

        if target_type not in supported_types:  # Access supported types statically
            raise ValueError(f"Unsupported target type for conversion: {target_type}")

        # First, evaluate the inner expression without a target type
        source_value = self._evaluate_cached(expr_to_convert, None)
        source_type = _get_python_type_name(
            source_value
        )  # Get type from evaluated value

        # Now, perform the explicit conversion
        converted_value = self._convert_type(source_value, source_type, target_type)

        # Finally, convert to the overall expected type if one was provided
        return self._convert_type(converted_value, target_type, expected_type)

    def _evaluate_string_concatenation(self, expression):
        """
        Evaluates string concatenation expressions (+ operator),
        respecting variables, literals, quotes, and basic nesting.
        """
        if self.debug:
            print(f"  String concat executing on: '{expression}'")
            # Show the specific variables context this evaluator instance is using
            print(
                f"  String concat evaluator vars ID: {id(self.variables)}, Content: {self.variables}"
            )

        result = ""
        parts = []
        idx = 0
        start = 0
        in_quotes = None  # Track the type of quote (' or ") currently active
        nesting_depth = 0  # Track depth of brackets '[]' and parentheses '()'

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

        # Add the final part of the expression (after the last '+' or if no '+')
        parts.append(expression[start:].strip())

        if self.debug:
            print(f"  String concat parts identified: {parts}")

        # Handle case where splitting resulted in empty list (e.g., expression was just "+")
        if not parts:
            # Depending on desired behavior, could return "" or raise error.
            # Returning "" seems reasonable for an empty/invalid concat expression.
            return ""

        for part in parts:
            part = part.strip()  # Ensure part itself has no leading/trailing whitespace
            if not part:
                # Skip empty parts that might result from splitting (e.g., "a + + b")
                continue

            if self.debug:
                print(f"      String concat evaluating part: '{part}'")

            try:
                # Evaluate the part recursively using the *same* evaluator instance,
                # ensuring it uses the correct variable scope (self.variables).
                # Evaluate expecting *any* type initially.
                value = self._evaluate_cached(part, None)

                if self.debug:
                    print(
                        f"      Part '{part}' evaluated to: {repr(value)} (type: {_get_python_type_name(value)})"
                    )

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

        if self.debug:
            print(f"  String concat final result: '{result}'")

        return result

    def _evaluate_list_literal(self, match):
        """Evaluates list literal expressions like [1, 2, "hello", x] using ast.literal_eval"""
        list_str = match.group(0)

        if self.debug:
            print(f"  Evaluating list literal: {list_str}")

        if list_str == "[]":
            return []

        # Need to replace variables within the list string before using literal_eval
        # This is complex and potentially unsafe if not done carefully.
        # A simple substitution might work for basic cases, but fails with nested structures or strings containing variable names.

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
                value = self._evaluate_cached(elem_expr, None)  # Evaluate element
                result_list.append(value)
            except ValueError as e:
                raise ValueError(f"Invalid list element expression: '{elem_expr}'. {e}")

        if self.debug:
            print(f"  Evaluated list literal result: {result_list}")
        return result_list

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
        value = self._evaluate_cached(value_expr, None)

        # Evaluate extra expression (position or replacement value) if present
        extra_value = None
        if extra_expr:
            extra_value = self._evaluate_cached(extra_expr, None)

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
        index_val = self._evaluate_cached(index_expression.strip(), "int")
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

        value_to_find = self._evaluate_cached(value_expression.strip(), None)

        if self.debug:
            print(
                f"  Finding value '{value_to_find}' in {self.variables[list_name][1]} '{list_name}'"
            )

        collection = self.variables[list_name][0]

        try:
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

    def _evaluate_boolean_negation(self, match):
        """Handles prefix boolean negation '~@ expr'"""
        sub_expr = match.group(1).strip()
        # Evaluate the sub-expression, expecting a boolean
        value = self._evaluate_cached(sub_expr, "bool")
        return not value

    def _evaluate_boolean_expression_infix(self, match):
        """Evaluates infix boolean expressions (e.g., a && b, x < 5)."""
        left_expression, operator, right_expression = match.groups()

        # Evaluate operands *without* assuming a type initially
        left_val = self._evaluate_cached(left_expression.strip(), None)
        right_val = self._evaluate_cached(right_expression.strip(), None)

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
