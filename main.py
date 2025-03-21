import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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


@dataclass
class Function:
    name: str
    params: List[tuple[str, str]]  # List of (param_name, param_type)
    body: List[str]
    return_type: str


class ExpressionEvaluator:
    def __init__(self, variables, debug=False):
        self.variables = variables
        self.debug = debug
        self.operators = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y if y != 0 else float("inf"),
            "%": lambda x, y: x % y if y != 0 else float("inf"),
        }
        self.type_conversions = {
            ("int", "float"): float,
            ("float", "int"): int,
            ("int", "bool"): bool,
            ("string", "list"): lambda x: [str(c) for c in x],
            ("any", "string"): str,
            ("int", "string"): str,
            ("float", "string"): str,
            ("bool", "string"): str,
            ("list", "string"): lambda x: str(x),
            ("string", "int"): lambda x: int(float(x))
            if x.replace(".", "").isdigit()
            else 0,
            ("string", "float"): lambda x: float(x)
            if x.replace(".", "").isdigit()
            else 0.0,
            ("void", "any"): lambda _: None,
            ("any", "void"): lambda x: x,
        }

    def evaluate(self, expression, expected_type):
        """Main evaluation method that routes to appropriate sub-evaluators"""
        if self.debug:
            print(
                f"Evaluating expression: '{expression}' with expected type: {expected_type}"
            )
        if expression in self.variables:
            value, var_type = self.variables[expression]
            if expected_type is None or var_type == expected_type:
                return value
            return self._convert_type(value, var_type, expected_type)
        # Handle string concatenation first if the expression contains '+'
        if "+" in expression and (expected_type == "string" or expected_type is None):
            try:
                result = self._evaluate_string_concatenation(expression)
                if result is not None:
                    return result
            except ValueError:
                pass

        # Then try other evaluation methods
        evaluators = [
            (r"([a-zA-Z_]\w*)\s*:>\s*(\w+)", self._evaluate_type_conversion),
            (r"\[.+\]", self._evaluate_list_literal),
            (r"(.+)\s*([<>]=?|&&|&\$\$&)\s*(.+)", self._evaluate_boolean_expression),
            (
                r'([a-zA-Z_]\w*)\s+("[^"]*"|\d+)\s+\[(a|r|n|p|P)\](?:\s+("[^"]*"|\d+))?',
                self._evaluate_list_operation,
            ),
            (r"([a-zA-Z_]\w*)\s+\[l\]", self._evaluate_list_length),
            (r"([a-zA-Z_]\w*)\s+\[i\]\s+(.+)", self._evaluate_list_indexing),
        ]

        for pattern, evaluator in evaluators:
            match = re.match(pattern, expression)
            if match:
                return evaluator(match, expected_type)

        # Try RPN evaluation for numeric expressions
        if expected_type in ("int", "float", None):
            try:
                result = self._evaluate_rpn(expression)
                if expected_type == "int":
                    return int(result)
                return result
            except ValueError:
                pass
        # string literal

        literal_result = self._evaluate_literal(expression, expected_type)
        if literal_result is not None:
            return literal_result

        raise ValueError(f"Unable to evaluate expression: {expression}")

    def _evaluate_literal(self, expression, expected_type):
        """Evaluates literal values with automatic type conversion"""
        # String literals
        if expression.startswith('"') and expression.endswith('"'):
            value = expression[1:-1]
            return self._convert_type(value, "string", expected_type)

        # Boolean literals
        if expression.lower() in ("true", "false"):
            value = expression.lower() == "true"
            return self._convert_type(value, "bool", expected_type)

        # Numeric literals
        if re.match(r"^-?\d+(\.\d+)?$", expression):
            if "." in expression:
                value = float(expression)
                return self._convert_type(value, "float", expected_type)
            value = int(expression)
            return self._convert_type(value, "int", expected_type)

        return None

    def _convert_type(self, value, source_type, target_type):
        """Enhanced type conversion system"""
        if source_type == target_type or target_type is None:
            return value

        key = (source_type, target_type)
        if key in self.type_conversions:
            try:
                return self.type_conversions[key](value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot convert {value} from {source_type} to {target_type}"
                )

        raise ValueError(
            f"Unsupported type conversion from {source_type} to {target_type}"
        )

    def _evaluate_string_concatenation(self, expression):
        """Evaluates string concatenation expressions."""
        print(f"String concat vars {self.variables}")
        parts = re.split(r"\s*\+\s*", expression)
        result = ""
        for part in parts:
            part = part.strip()
            # Handle string literals
            if part.startswith('"') and part.endswith('"'):
                result += part[1:-1]
            # Handle variables
            elif part in self.variables:
                value, var_type = self.variables[part]
                result += str(value)
            # Handle other expressions
            else:
                try:
                    value = self.evaluate(part, None)
                    result += str(value)
                except ValueError:
                    raise ValueError(f"Invalid part in string concatenation: {part}")

        return result

    def _is_list_type(self, list_name):
        if list_name not in self.variables or (
            self.variables[list_name][1] != "list"
            and self.variables[list_name][1] != "string"
        ):
            return False
        return True

    def _evaluate_list_operation(self, match, expected_type=None):
        """Evaluates list append/remove/insert/replace operations."""
        list_name, value_str, operation, extra = match.groups()
        if not self._is_list_type(list_name):
            raise ValueError(f"'{list_name}' is not a defined list or string.")

        # Get the value and type of the list/string
        list_value, list_type = self.variables[list_name]

        # Handle string literals in quotes
        if value_str.startswith('"') and value_str.endswith('"'):
            value = value_str[1:-1]  # Remove quotes
        else:
            # Evaluate the value expression
            value = self.evaluate(value_str, None)

        if operation == "a":  # Append
            if list_type == "list":
                list_value.append(value)
            else:  # string
                list_value += value
            return list_value
        elif operation == "r":  # Remove
            if list_type == "list":
                if value in list_value:
                    list_value.remove(value)
            else:  # string
                list_value = list_value.replace(value, "")
            return list_value
        elif operation == "n":  # Insert
            if extra is None:
                raise ValueError("Insert operation requires a position")
            try:
                position = int(extra.strip())
                if list_type == "list":
                    # Convert to 0-based indexing
                    list_value.insert(position - 1, value)
                else:  # string
                    list_value = (
                        list_value[: position - 1] + value + list_value[position - 1 :]
                    )
                return list_value
            except (ValueError, TypeError):
                raise ValueError("Invalid position for insert operation")
        elif operation == "p" or operation == "P":  # Replace/Replace All
            if extra is None:
                raise ValueError("Replace operation requires a new value")
            # Handle string literals in quotes for the replacement value
            if extra.strip().startswith('"') and extra.strip().endswith('"'):
                new_value = extra.strip()[1:-1]  # Remove quotes
            else:
                new_value = extra.strip()

            if list_type == "list":
                try:
                    index = list_value.index(value)
                    list_value[index] = new_value
                except ValueError:
                    raise ValueError(f"Value {value} not found in list {list_name}")
            else:  # string
                if operation == "p":
                    # Replace only first occurrence
                    list_value = list_value.replace(value, new_value, 1)
                else:
                    list_value = list_value.replace(
                        value, new_value
                    )  # Replace all occurrences
            return list_value
        else:
            raise ValueError(f"Invalid list operation: {operation}")

    def _evaluate_list_length(self, match, expected_type="int"):
        list_name = match.group(1)
        if not self._is_list_type(list_name):
            raise ValueError(f"'{list_name}' is not a defined list.")
        length = len(self.variables[list_name][0])
        if self.debug:
            print(f" List length result: {length}")
        return length

    def _evaluate_list_indexing(self, match, expected_type=None):
        """Evaluates list indexing operations."""
        list_name, index_expression = match.groups()
        if not self._is_list_type(list_name):
            raise ValueError(f"'{list_name}' is not a defined list.")
        index = self.evaluate(index_expression.strip(), "int")
        if not isinstance(index, int):
            raise ValueError(
                f"Index must evaluate to an integer. Got: {type(index).__name__}"
            )
        try:
            return self.variables[list_name][0][index - 1]
        except IndexError:
            print(f"Warning: Index {index} out of range for list {list_name}")
            return None  # Or raise the exception, depending on the desired behavior

    def _evaluate_type_conversion(self, match, expected_type=None):
        """Evaluates type conversion expressions."""
        var_name, target_type = match.groups()
        if var_name not in self.variables:
            # Try to evaluate the expression first
            try:
                value = self.evaluate(var_name, None)
                return self._convert_type(
                    value, type(value).__name__.lower(), target_type
                )
            except ValueError:
                raise ValueError(f"Variable or expression '{var_name}' not defined.")

        source_value, source_type = self.variables[var_name]
        return self._convert_type(source_value, source_type, target_type)

    def _evaluate_rpn(self, expression):
        """Evaluates RPN arithmetic expressions."""
        stack = []
        tokens = expression.split()

        for token in tokens:
            # Check for numeric literals, accounting for negative signs and decimals
            if re.match(r"^-?\d+(\.\d*)?$", token):  # Improved regex
                try:
                    if "." in token:
                        stack.append(float(token))
                    else:
                        stack.append(int(token))
                except ValueError:  # Could happen if token is something like "1.a"
                    raise ValueError(f"Invalid number literal: {token}")
            elif token in self.variables:
                var_value, var_type = self.variables[token]
                if var_type in ("int", "float"):
                    stack.append(var_value)
                else:
                    raise ValueError(
                        f"Variable '{token}' is not numeric (type: {var_type})"
                    )
            elif token in ("+", "-", "*", "/", "%"):
                if len(stack) < 2:
                    raise ValueError(f"Insufficient operands for operator '{token}'")
                try:
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                except IndexError:
                    raise ValueError("Not enough operands on stack")

                try:
                    if token == "+":
                        stack.append(operand1 + operand2)
                    elif token == "-":
                        stack.append(operand1 - operand2)
                    elif token == "*":
                        stack.append(operand1 * operand2)
                    elif token == "/":
                        if operand2 == 0:
                            raise ZeroDivisionError("Division by zero")
                        stack.append(operand1 / operand2)
                    elif token == "%":
                        if operand2 == 0:
                            raise ZeroDivisionError("Modulo by zero")
                        stack.append(operand1 % operand2)
                except TypeError:
                    raise ValueError("Invalid operand types for operator")
            else:
                raise ValueError(f"Invalid RPN token: {token}")

        if len(stack) != 1:
            raise ValueError("Invalid RPN expression")

        return stack.pop()

    def _evaluate_boolean_expression(self, match, expected_type="bool"):
        """Evaluates non-RPN boolean expressions."""
        if isinstance(match, str):
            expression = match
        else:
            left_expression, operator, right_expression = match.groups()
            expression = f"{left_expression} {operator} {right_expression}"

        if expression.lower() == "true":
            return True
        elif expression.lower() == "false":
            return False

        match = re.match(r"(.+)\s*([<>]=?|&&|&\$\$&)\s*(.+)", expression)
        if not match:
            raise ValueError(f"Invalid boolean expression: {expression}")

        left_expression, operator, right_expression = match.groups()
        left_val = self.evaluate(left_expression.strip(), None)
        right_val = self.evaluate(right_expression.strip(), None)

        left_type = type(left_val).__name__
        right_type = type(right_val).__name__

        if operator in ("&&", "&$$&"):
            if left_type != right_type and not (
                left_type in ("int", "float") and right_type in ("int", "float")
            ):
                raise ValueError(
                    f"Type mismatch: Cannot compare {left_type} with {right_type}"
                )
        elif operator in ("<", ">", "<=", ">="):
            if not (
                (left_type in ("int", "float") and right_type in ("int", "float"))
                or (left_type == right_type == "string")
            ):
                raise ValueError(
                    f"Type mismatch: Cannot compare {left_type} with {right_type} using {operator}"
                )

        if operator == "&&":
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) == float(right_val)
            return str(left_val) == str(right_val)
        elif operator == "&$$&":
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) != float(right_val)
            return str(left_val) != str(right_val)
        elif operator == "<":
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) < float(right_val)
            return str(left_val) < str(right_val)
        elif operator == ">":
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) > float(right_val)
            return str(left_val) > str(right_val)
        elif operator == "<=":
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) <= float(right_val)
            return str(left_val) <= str(right_val)
        elif operator == ">=":
            if isinstance(left_val, (int, float)) and isinstance(
                right_val, (int, float)
            ):
                return float(left_val) >= float(right_val)
            return str(left_val) >= str(right_val)

    def _evaluate_list_literal(self, match, expected_type):
        """Evaluates list literal expressions like [1, 2, "hello", x]"""
        if expected_type not in ("list", "string", None):
            raise ValueError(f"Cannot evaluate list literal as type {expected_type}")

        # Extract the content between brackets
        content = match.group(0)[1:-1].strip()

        # Handle empty list
        if not content:
            return [] if expected_type != "string" else ""

        # Split by commas, handling nested structures
        elements = []
        current = ""
        bracket_count = 0
        quote_char = None

        for char in content:
            if char in ('"', "'") and not quote_char:
                quote_char = char
            elif char == quote_char:
                quote_char = None
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            elif char == "," and bracket_count == 0 and not quote_char:
                elements.append(current.strip())
                current = ""
                continue
            current += char

        if current:
            elements.append(current.strip())

        # Evaluate each element
        result = []
        for element in elements:
            try:
                # Recursively evaluate each element
                value = self.evaluate(element, None)
                result.append(value)
            except ValueError as e:
                raise ValueError(f"Invalid list element: {element}. {str(e)}")

        if expected_type == "string":
            return "".join(str(x) for x in result)
        return result

    def _evaluate_literals_and_variables(self, expression, expected_type):
        if expression.startswith('"') and expression.endswith('"'):
            return expression[1:-1]
        elif expression.startswith("[") and expression.endswith("]"):
            try:
                # Handle empty list
                if expression == "[]":
                    return []
                elements = [x.strip() for x in expression[1:-1].split(",") if x.strip()]
                if self.debug:
                    print(f" List literal: {elements}")
                return elements
            except ValueError:
                raise ValueError(f"Invalid list literal: {expression}")
        elif expression.lower() == "true":
            return True
        elif expression.lower() == "false":
            return False
        elif expression.isdigit():
            return int(expression)
        elif re.match(r"^[+-]?\d+(\.\d+)?$", expression):  # improved float regex
            return float(expression)
        elif expression in self.variables:
            var_value, _ = self.variables[expression]
            return var_value
        else:
            raise ValueError(f"Invalid expression or undefined variable: {expression}")


class Interpreter:
    def __init__(self, debug=False, debug_show=False):
        self.variables = {}
        self.functions = {}  # Store defined functions
        self.supported_types = ["int", "float", "string", "list", "bool", "set", "void"]
        self.expression_evaluator = ExpressionEvaluator(self.variables, debug)
        self.debug = debug
        self.debug_show = debug_show
        self.return_value = None  # Store function return values

    def run(self, filepath):
        """Runs the interpreter on the given script file."""
        try:
            with open(filepath, "r") as file:
                lines = file.readlines()
                self.execute_lines(lines)
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_repl(self):
        """Run the interpreter in REPL (Read-Eval-Print Loop) mode."""
        print("Pepper Programming Language REPL")
        print("Type 'exit' to quit, 'help' for commands")
        print("Variables:", end=" ")

        while True:
            try:
                # Show current variables as context
                if self.variables:
                    vars_str = ", ".join(
                        f"{k}: {v[1]}" for k, v in self.variables.items()
                    )
                    print(f"\nVariables: {vars_str}")

                # Get input
                line = input(">>> ").strip()

                # Handle special commands
                if line.lower() == "exit":
                    break
                elif line.lower() == "help":
                    self._show_help()
                    continue
                elif line.lower() == "clear":
                    self.variables.clear()
                    print("All variables cleared")
                    continue
                elif not line:
                    continue

                # Execute the line
                self.execute_lines([line])

            except Exception as e:
                print(f"Error: {str(e)}")

    def _show_help(self):
        """Show help information for REPL mode."""
        help_text = """
Available Commands:
    LET x: type = value    - Declare a variable
    REAS x = value        - Reassign a variable
    SHOW expression       - Display a value
    exit                  - Exit REPL
    help                  - Show this help
    clear                 - Clear all variables

Types:
    int     - Integer numbers
    float   - Decimal numbers
    string  - Text in quotes
    bool    - True/False values
    list    - Lists of values

List Operations:
    [a]     - Append
    [r]     - Remove
    [n]     - Insert at position
    [p]     - Replace first occurrence
    [P]     - Replace all occurrences
    [l]     - Get length
    [i]     - Get item at index
"""
        print(help_text)

    def execute_lines(self, lines):
        """Executes the lines of code."""
        line_number = 0
        skip_lines = 0
        inside_if = False
        loop_stack = []

        while line_number < len(lines):
            line = lines[line_number].split('%%')[0].strip()

            if skip_lines > 0:
                skip_lines -= 1
                if self.debug:
                    print(f"Skipping line {line_number + 1}: {line}")
                line_number += 1
                continue

            if not line or line.startswith("%%"):
                if self.debug:
                    print(f"Skipping line {line_number + 1}: {line}")
                line_number += 1
                continue

            if self.debug:
                print(f"Processing line {line_number + 1}: {line}")

            try:
                if "::(" in line:  # Function declaration
                    skip_lines = self.handle_function_declaration(lines, line_number)
                    line_number += skip_lines
                elif "|>" in line:
                    self.handle_function_call(line)
                elif line.startswith(IF_KEYWORD):
                    inside_if = True
                    skip_lines = self.handle_if(line, lines, line_number)
                elif line.startswith(FOR_KEYWORD):
                    loop_stack.append((FOR_KEYWORD, line_number))
                    line_number = self.handle_for(line, lines, line_number)
                    continue
                elif line.startswith(WHILE_KEYWORD):
                    loop_stack.append((WHILE_KEYWORD, line_number))
                    line_number = self.handle_while(line, lines, line_number)
                    continue
                elif line.startswith(LOOP_END_KEYWORD):
                    if not loop_stack:
                        raise ValueError(
                            f"{LOOP_END_KEYWORD} without matching {FOR_KEYWORD} or {WHILE_KEYWORD} on line {line_number + 1}"
                        )
                    loop_type, start_line = loop_stack.pop()

                    if loop_type == FOR_KEYWORD:
                        pass  # For loop, simply go to next line after 'LOOP_END'
                    elif loop_type == WHILE_KEYWORD:
                        line_number = start_line
                        continue
                elif line.startswith(LET_KEYWORD):
                    if (inside_if and skip_lines == 0) or not inside_if:
                        self.handle_let(line)
                elif line.startswith(REAS_KEYWORD):
                    if (inside_if and skip_lines == 0) or not inside_if:
                        self.handle_reas(line)
                elif line.startswith(SHOW_KEYWORD):
                    if (inside_if and skip_lines == 0) or not inside_if:
                        self.handle_show(line)
                elif (
                    line.startswith(ELIF_KEYWORD)
                    or line.startswith(ELSE_KEYWORD)
                    or line.startswith(END_KEYWORD)
                ):
                    if not inside_if:
                        raise ValueError(
                            f"Invalid command: {line}. {ELIF_KEYWORD}, {ELSE_KEYWORD}, and {END_KEYWORD} are only valid inside an {IF_KEYWORD} block"
                        )
                else:
                    raise ValueError(f"Invalid command: {line}")

            except ValueError as e:
                print(f"Error on line {line_number + 1}: {e}")
                return  # Or perhaps continue, depending on desired error handling

            if self.debug:
                print(
                    f"Current variables: {self.variables}\nCurrent functions: {self.functions}"
                )
                print("-" * 20)
            line_number += 1

    def handle_for(self, initial_for_line, lines, start_index):
        """Handles FOR loops. Syntax: FOR var FROM start TO end DO ... LOOP_END"""
        match = re.match(
            r"FOR\s+([a-zA-Z_]\w*)\s+FROM\s+(.+)\s+TO\s+(.+)\s+DO", initial_for_line
        )
        if not match:
            raise ValueError(f"Invalid FOR statement: {initial_for_line}")

        var_name, start_expr, end_expr = match.groups()

        start_val = self.expression_evaluator.evaluate(start_expr, "int")
        end_val = self.expression_evaluator.evaluate(end_expr, "int")

        # Initialize loop variable
        self.variables[var_name] = (start_val, "int")

        current_index = start_index + 1
        while self.variables[var_name][0] <= end_val:
            while current_index < len(lines):
                current_line = lines[current_index].strip()
                if current_line.startswith("LOOP_END"):
                    break  # Exit inner loop when LOOP_END is encountered
                elif current_line.startswith("FOR"):
                    current_index = self.handle_for(current_line, lines, current_index)
                    continue
                elif current_line.startswith("WHILE"):
                    current_index = self.handle_while(
                        current_line, lines, current_index
                    )
                    continue
                else:
                    # Execute lines within the loop
                    self.execute_line(current_line)
                current_index += 1

            # Increment loop variable
            self.variables[var_name] = (self.variables[var_name][0] + 1, "int")
            current_index = start_index + 1

        current_index = start_index + 1
        while current_index < len(lines):
            if lines[current_index].strip().startswith("LOOP_END"):
                return current_index
            current_index += 1
        raise ValueError(
            f"FOR loop starting on line {start_index + 1} has no matching LOOP_END"
        )

    def handle_while(self, initial_while_line, lines, start_index):
        """Handles WHILE loops.  Syntax: WHILE condition DO ... LOOP_END"""
        match = re.match(r"WHILE\s+(.+)\s+DO", initial_while_line)
        if not match:
            raise ValueError(f"Invalid WHILE statement: {initial_while_line}")

        condition_str = match.group(1)
        current_index = start_index + 1

        while self.expression_evaluator.evaluate(condition_str, "bool"):
            while current_index < len(lines):
                current_line = lines[current_index].strip()
                if current_line.startswith("LOOP_END"):
                    break  # Exit the loop
                elif current_line.startswith("FOR"):
                    current_index = self.handle_for(current_line, lines, current_index)
                    continue
                elif current_line.startswith("WHILE"):
                    current_index = self.handle_while(
                        current_line, lines, current_index
                    )
                    continue
                else:
                    self.execute_line(current_line)
                current_index += 1
            current_index = start_index + 1

        current_index = start_index + 1
        while current_index < len(lines):
            if lines[current_index].strip().startswith("LOOP_END"):
                return current_index  # Return index of LOOP_END
            current_index += 1
        raise ValueError(
            f"WHILE loop starting on line {start_index + 1} has no matching LOOP_END"
        )

    def handle_if(self, initial_if_line, lines, start_index):
        """Handles IF-ELIF-ELSE-END blocks."""
        match = re.match(r"IF\s+(.+)\s+DO", initial_if_line)
        if not match:
            raise ValueError(f"Invalid IF statement: {initial_if_line}")
        condition_str = match.group(1)
        condition_result = self.expression_evaluator.evaluate(condition_str, "bool")

        block_executed = False
        lines_to_skip = 0
        current_index = start_index + 1

        # Skip the IF block if condition is false
        if not condition_result:
            nesting_level = 0
            while current_index < len(lines):
                current_line = lines[current_index].strip()
                if current_line.startswith("IF"):
                    nesting_level += 1
                elif current_line.startswith("END"):
                    if nesting_level == 0:
                        break
                    nesting_level -= 1
                elif nesting_level == 0 and (
                    current_line.startswith("ELIF") or current_line.startswith("ELSE")
                ):
                    break
                current_index += 1
                lines_to_skip += 1

        # Execute the IF block if condition is true
        if condition_result:
            block_executed = True
            while current_index < len(lines):
                current_line = lines[current_index].strip()
                if (
                    current_line.startswith("ELIF")
                    or current_line.startswith("ELSE")
                    or current_line.startswith("END")
                ):
                    break
                elif current_line.startswith("IF"):
                    lines_to_skip_nested_if = self.handle_if(
                        current_line, lines, current_index
                    )
                    current_index += lines_to_skip_nested_if + 1
                    lines_to_skip += lines_to_skip_nested_if + 1
                else:
                    self.execute_line(current_line)
                    current_index += 1
                    lines_to_skip += 1

        # Process ELIF blocks if no block has been executed
        while not block_executed and current_index < len(lines):
            current_line = lines[current_index].strip()

            if current_line.startswith("ELIF"):
                match = re.match(r"ELIF\s+(.+)\s+DO", current_line)
                if not match:
                    raise ValueError(f"Invalid ELIF statement: {current_line}")
                condition_str = match.group(1)
                condition_result = self.expression_evaluator.evaluate(
                    condition_str, "bool"
                )
                current_index += 1
                lines_to_skip += 1

                if condition_result:
                    block_executed = True
                    while current_index < len(lines):
                        current_line = lines[current_index].strip()
                        if (
                            current_line.startswith("ELIF")
                            or current_line.startswith("ELSE")
                            or current_line.startswith("END")
                        ):
                            break
                        elif current_line.startswith("IF"):
                            lines_to_skip_nested_if = self.handle_if(
                                current_line, lines, current_index
                            )
                            current_index += lines_to_skip_nested_if + 1
                            lines_to_skip += lines_to_skip_nested_if + 1
                        else:
                            self.execute_line(current_line)
                            current_index += 1
                            lines_to_skip += 1
                else:
                    # Skip this ELIF block
                    nesting_level = 0
                    while current_index < len(lines):
                        current_line = lines[current_index].strip()
                        if current_line.startswith("IF"):
                            nesting_level += 1
                        elif current_line.startswith("END"):
                            if nesting_level == 0:
                                break
                            nesting_level -= 1
                        elif nesting_level == 0 and (
                            current_line.startswith("ELIF")
                            or current_line.startswith("ELSE")
                        ):
                            break
                        current_index += 1
                        lines_to_skip += 1
            elif current_line.startswith("ELSE"):
                block_executed = True
                current_index += 1
                lines_to_skip += 1
                while current_index < len(lines):
                    current_line = lines[current_index].strip()
                    if current_line.startswith("END"):
                        break
                    elif current_line.startswith("IF"):
                        lines_to_skip_nested_if = self.handle_if(
                            current_line, lines, current_index
                        )
                        current_index += lines_to_skip_nested_if + 1
                        lines_to_skip += lines_to_skip_nested_if + 1
                    else:
                        self.execute_line(current_line)
                        current_index += 1
                        lines_to_skip += 1
            elif current_line.startswith("END"):
                break
            else:
                current_index += 1
                lines_to_skip += 1

        # Skip to the END statement
        while current_index < len(lines):
            current_line = lines[current_index].strip()
            if current_line.startswith("END"):
                break
            current_index += 1
            lines_to_skip += 1

        # Skip the END statement itself
        if current_index < len(lines) and lines[current_index].strip().startswith(
            "END"
        ):
            current_index += 1
            lines_to_skip += 1

        return lines_to_skip

    def execute_line(self, line):
        """Executes a single line of code."""
        if line.startswith(LET_KEYWORD):
            self.handle_let(line)
        elif "|>" in line:  # Function call
            print("found |>")
            return self.handle_function_call(line)
        elif line.startswith(SHOW_KEYWORD):
            self.handle_show(line)
        elif line.startswith(REAS_KEYWORD):
            self.handle_reas(line)
        elif "::(" in line:  # Function declaration
            return self.handle_function_declaration([line], 0)
        else:
            raise ValueError(f"Invalid command: {line}")

    def handle_let(self, line):
        """Handles LET statements."""
        match = re.match(r"LET\s+([a-zA-Z_]\w*)\s*:\s*(\w+)\s*=\s*(.+)", line)
        if not match:
            raise ValueError(f"Invalid LET statement: {line}")

        var_name, var_type, expression = match.groups()
        if self.debug:
            print(
                f"  Parsed LET: var_name={var_name}, var_type={var_type}, expression={expression}"
            )

        if var_type not in self.supported_types:
            raise ValueError(f"Unsupported data type: {var_type}")

        value = self.expression_evaluator.evaluate(expression, var_type)
        if var_name in self.variables:
            print(
                f"Error: Variable {var_name} already exists. Use REAS to reassign a variable"
            )
            return
        self.variables[var_name] = (value, var_type)
        if self.debug:
            print(f"Declared variable: {var_name} = {value} (type: {var_type})")

    def handle_reas(self, line):
        """Handles REAS statements."""
        match = re.match(r"REAS\s+([a-zA-Z_]\w*)\s*=\s*(.+)", line)
        if not match:
            raise ValueError(f"Invalid REAS statement: {line}")

        var_name, expression = match.groups()

        if var_name not in self.variables:
            print(
                f"Error: Variable {var_name} does not exist. Use LET to create a variable"
            )
            return
        var_type = self.variables[var_name][1]  # Get the original type
        if self.debug:
            print(f"  Parsed REAS: var_name={var_name}, expression={expression}")

        value = self.expression_evaluator.evaluate(expression, var_type)
        self.variables[var_name] = (value, var_type)
        if self.debug:
            print(f"Reassigned variable: {var_name} = {value} (type: {var_type})")

    def handle_show(self, line):
        """Handles SHOW statements."""
        match = re.match(r"SHOW\((.+)\)", line)
        if not match:
            raise ValueError(f"Invalid SHOW statement: {line}")

        expression = match.group(1).strip()
        if self.debug:
            print(f"  Parsed SHOW: expression={expression}")
            print(f"  {self.variables}")
        try:
            # Check if the expression involves string concatenation
            if "+" in expression and expression.replace(" ", "").startswith('"'):
                value = self.expression_evaluator._evaluate_string_concatenation(
                    expression
                )
            else:
                value = self.expression_evaluator.evaluate(expression, None)
            if self.debug or self.debug_show:
                print(f"SHOW: {value}")
            else:
                print(value)
        except ValueError as e:
            print(f"Error in SHOW: {e}")

    def handle_function_declaration(self, lines: List[str], start_index: int) -> int:
        """Handles function declaration and returns number of lines to skip"""
        # Parse function declaration line
        decl_line = lines[start_index].strip()
        match = re.match(r"([a-zA-Z_]\w*)::\((.*?)\)->", decl_line)
        if not match:
            raise ValueError(f"Invalid function declaration: {decl_line}")

        func_name, params_str = match.groups()

        # Parse parameters
        params = []
        if params_str.strip():
            for param in params_str.split(","):
                param = param.strip()
                param_match = re.match(r"([a-zA-Z_]\w*):(\w+)", param)
                if not param_match:
                    raise ValueError(f"Invalid parameter declaration: {param}")
                param_name, param_type = param_match.groups()
                params.append((param_name, param_type))

        print(params)
        # Collect function body
        body = []
        current_index = start_index + 1
        while current_index < len(lines):
            line = lines[current_index].strip()
            if line.startswith("<-"):
                # Parse return type
                return_type = line[2:].strip()
                if return_type not in self.supported_types:
                    raise ValueError(f"Unsupported return type: {return_type}")

                # Store function
                self.functions[func_name] = Function(
                    func_name, params, body, return_type
                )
                print(len(body))
                return len(body)

            body.append(line)
            current_index += 1

        raise ValueError(f"Function {func_name} has no return type declaration")

    def handle_function_call(self, line: str) -> Any:
        """Handles function calls with the |> operator"""
        if self.debug:
            print(f"  Parsed function call: {line}")
        match = re.match(r"(\(.*?\)|_)\s*\|>\s*([a-zA-Z_]\w*)", line)
        if not match:
            raise ValueError(f"Invalid function call: {line}")

        args_str, func_name = match.groups()
        print(match.groups())

        if func_name not in self.functions:
            raise ValueError(f"Undefined function: {func_name}")

        func = self.functions[func_name]

        # Parse arguments
        args = []
        if args_str != "_" and args_str.strip():
            args_str = args_str[1:-1]
            args = [arg.strip() for arg in args_str.split(",")]

        if len(args) != len(func.params):
            raise ValueError(
                f"Function {func_name} expects {len(func.params)} arguments, got {len(args)}"
            )

        # Create new scope for function variables
        old_variables = self.variables.copy()
        self.variables = {}
        self.variables.update(old_variables)

        # Evaluate and bind arguments to parameters
        for (param_name, param_type), arg in zip(func.params, args):
            print(arg, param_type)
            value = self.expression_evaluator.evaluate(arg, param_type)
            print(value)
            self.variables[param_name] = (value, param_type)
        print(self.variables)

        # Execute function body
        self.return_value = None
        for line in func.body:
            if line.startswith("RETURN "):
                expr = line[7:].strip()
                self.return_value = self.expression_evaluator.evaluate(
                    expr, func.return_type
                )
                break
            else:
                print()
                self.execute_line(line)

        # Restore original scope
        # self.variables = old_variables

        return self.return_value


def main():
    parser = argparse.ArgumentParser(description="A simple code interpreter.")
    parser.add_argument("filepath", nargs="?", help="The path to the script file.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument(
        "-s", "--debug_show", action="store_true", help="Enable debug show mode."
    )
    args = parser.parse_args()

    interpreter = Interpreter(args.debug, args.debug_show)

    if args.filepath:
        interpreter.run(args.filepath)
    else:
        interpreter.run_repl()


if __name__ == "__main__":
    main()

