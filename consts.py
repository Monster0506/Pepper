from typing import List, Optional, Match
import re
from dataclasses import dataclass

supported_types = [
    "int",
    "float",
    "string",
    "list",
    "bool",
    "any",
    "void",
]  # Added 'any'


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


REGEX = {
    "int": re.compile(r"-?\d+"),
    "float": re.compile(r"-?(\d+(\.\d*)?|\.\d+)"),
    "numeric": re.compile(
        r"-?(\d+(\.\d*)?|\.\d+)|-?\d+"
    ),  # Combined version (optional)
    "var_name": re.compile(r"[a-zA-Z_]\w*"),
    "input": re.compile(r'INPT\("([^"]*)"\)'),
    "type_conversion": re.compile(r"(.+?)\s*:>\s*(\w+)"),
    "list_literal": re.compile(r"\[.*\]", re.DOTALL),
    "list_operation": re.compile(
        r'([a-zA-Z_]\w*)\s+("[^"]*"|\S+)\s+\[(a|r|n|p|P)\](?:\s+("[^"]*"|\S+))?'
    ),
    "list_length": re.compile(r"([a-zA-Z_]\w*)\s+\[l\]"),
    "list_random": re.compile(r"([a-zA-Z_]\w*)\s+\[\?\]"),
    "list_index": re.compile(r"([a-zA-Z_]\w*)\s+\[i\]\s+(.+)"),
    "list_find": re.compile(r"([a-zA-Z_]\w*)\s+\[f\]\s+(.+)"),
    "function_pipe": re.compile(r"(\(.*?\)|_)\s*\|>\s*([a-zA-Z_]\w*)"),
    "bool_infix": re.compile(r"(.+?)\s*(@\$@|#\$#|&&|&\$\$&|[<>]=?)\s*(.+)"),
    "bool_negation": re.compile(r"~@\s+(.+)"),
    "stdlib_call": re.compile(
        r"([a-zA-Z_]\w*)\s+FROM\s+([a-zA-Z_]\w*)\s+(.*?)\s*(\(.*\)|\_)$"
    ),
    "string_literal": re.compile(r'\s*"((?:[^"\\]|\\.)*)"\s*'),
    "trivial_list_literal": re.compile(r"\s*\[(.*)\]\s*"),  # For quick null-match check
    "maybe_rpn": re.compile(r"[\d\.\s+\-*/%?]"),
    "has_quotes": re.compile(r'"'),
}


def matchEXPR(
    expression: str, name: str, search: bool = False
) -> Optional[re.Match[str]]:
    if not search:
        return REGEX[name].fullmatch(expression)
    return REGEX[name].search(expression)
