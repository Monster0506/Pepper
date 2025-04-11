from typing import List
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
