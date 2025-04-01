import random
import re
import math


class StandardLibrary:
    """Encapsulates standard library functions for the Pepper language, organized by namespace."""

    def __init__(self, evaluator):
        self._evaluator = evaluator
        self._debug = evaluator.debug
        # Nested dictionary for namespaced functions
        self.libraries = {
            "string": {
                "upper": self.string_upper,
                "lower": self.string_lower,
                "len": self.string_len,
                "trim": self.string_trim,
                "replace": self.string_replace,
                "split": self.string_split,
                "join": self.string_join,
                "contains": self.string_contains,
                "starts_with": self.string_starts_with,
                "ends_with": self.string_ends_with,
                "repeat": self.string_repeat,
                "reverse": self.string_reverse,
                "substr": self.string_substr,
                "pad_left": self.string_pad_left,
                "pad_right": self.string_pad_right,
                "capitalize": self.string_capitalize,
                "title": self.string_title,
            },
            "math": {
                "sqrt": self.math_sqrt,
                "pow": self.math_pow,
                "abs": self.math_abs,
                "floor": self.math_floor,
                "ceil": self.math_ceil,
                "round": self.math_round,
                "min": self.math_min,
                "max": self.math_max,
                "sin": self.math_sin,
                "cos": self.math_cos,
                "tan": self.math_tan,
                "asin": self.math_asin,
                "acos": self.math_acos,
                "atan": self.math_atan,
                "log": self.math_log,
                "log10": self.math_log10,
                "exp": self.math_exp,
                "pi": self.math_pi,
                "e": self.math_e,
                "factorial": self.math_factorial,
                "gcd": self.math_gcd,
                "lcm": self.math_lcm,
                "is_prime": self.math_is_prime,
                "degrees": self.math_degrees,
                "radians": self.math_radians,
            },
            "random": {
                "rnd": self.math_rnd,
                "choice": self.random_choice,
                "shuffle": self.random_shuffle,
                "sample": self.random_sample,
            },
            "type": {
                "is_int": self.type_is_int,
                "is_float": self.type_is_float,
                "is_string": self.type_is_string,
                "is_list": self.type_is_list,
                "is_bool": self.type_is_bool,
                "get": self.type_get_type,
                "to_int": self.type_to_int,
                "to_float": self.type_to_float,
                "to_string": self.type_to_string,
                "to_bool": self.type_to_bool,
            },
            "list": {
                "sort": self.list_sort,
                "reverse": self.list_reverse,
                "map": self.list_map,
                "filter": self.list_filter,
                "reduce": self.list_reduce,
                "find": self.list_find,
                "count": self.list_count,
                "sum": self.list_sum,
                "avg": self.list_avg,
                "zip": self.list_zip,
                "enumerate": self.list_enumerate,
                "slice": self.list_slice,
                "concat": self.list_concat,
                "unique": self.list_unique,
            },
            "time": {
                "now": self.time_now,
                "sleep": self.time_sleep,
                "format": self.time_format,
                "parse": self.time_parse,
            },
            "file": {
                "read": self.file_read,
                "write": self.file_write,
                "exists": self.file_exists,
                "delete": self.file_delete,
                "rename": self.file_rename,
                "size": self.file_size,
            },
            "system": {
                "args": self.system_args,
                "exit": self.system_exit,
                "env": self.system_env,
                "exec": self.system_exec,
            },
        }

    # Helper to get evaluator's convert function
    def _convert(self, value, source_type, target_type):
        try:
            return self._evaluator._convert_type(value, source_type, target_type)
        except Exception as e:
            raise ValueError(
                f"Stdlib conversion failed converting {value} from {source_type} to {target_type}: {e}"
            )
        # --- Additional Math Functions ---

    def math_asin(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.asin' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            if not -1 <= num_base <= 1:
                raise ValueError("'math.asin' argument must be between -1 and 1")
            return math.asin(num_base)
        except ValueError as e:
            raise ValueError(f"'math.asin' error: {e}")

    def math_acos(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.acos' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            if not -1 <= num_base <= 1:
                raise ValueError("'math.acos' argument must be between -1 and 1")
            return math.acos(num_base)
        except ValueError as e:
            raise ValueError(f"'math.acos' error: {e}")

    def math_atan(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.atan' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.atan(num_base)
        except ValueError:
            raise ValueError("'math.atan' requires a numeric base value")

    def math_log(self, base_value, base_type, args):
        if len(args) > 1:
            raise ValueError("'math.log' takes at most 1 argument (base)")
        try:
            num_base = self._convert(base_value, base_type, "float")
            if num_base <= 0:
                raise ValueError("'math.log' argument must be positive")
            if len(args) == 1:
                log_base = self._convert(
                    args[0], _get_python_type_name(args[0]), "float"
                )
                return math.log(num_base, log_base)
            return math.log(num_base)
        except ValueError as e:
            raise ValueError(f"'math.log' error: {e}")

    def math_log10(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.log10' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            if num_base <= 0:
                raise ValueError("'math.log10' argument must be positive")
            return math.log10(num_base)
        except ValueError as e:
            raise ValueError(f"'math.log10' error: {e}")

    def math_exp(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.exp' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.exp(num_base)
        except ValueError:
            raise ValueError("'math.exp' requires a numeric base value")

    def math_pi(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.pi' takes no arguments")
        return math.pi

    def math_e(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.e' takes no arguments")
        return math.e

    def math_factorial(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.factorial' takes no arguments")
        try:
            num = self._convert(base_value, base_type, "int")
            if num < 0:
                raise ValueError("'math.factorial' argument must be non-negative")
            return math.factorial(num)
        except ValueError as e:
            raise ValueError(f"'math.factorial' error: {e}")

    def math_gcd(self, base_value, base_type, args):
        if len(args) != 1:
            raise ValueError("'math.gcd' requires exactly one argument")
        try:
            a = self._convert(base_value, base_type, "int")
            b = self._convert(args[0], _get_python_type_name(args[0]), "int")
            return math.gcd(abs(a), abs(b))
        except ValueError:
            raise ValueError("'math.gcd' requires integer arguments")

    def math_lcm(self, base_value, base_type, args):
        if len(args) != 1:
            raise ValueError("'math.lcm' requires exactly one argument")
        try:
            a = self._convert(base_value, base_type, "int")
            b = self._convert(args[0], _get_python_type_name(args[0]), "int")
            return abs(a * b) // math.gcd(a, b)
        except ValueError:
            raise ValueError("'math.lcm' requires integer arguments")

    def math_is_prime(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.is_prime' takes no arguments")
        try:
            num = self._convert(base_value, base_type, "int")
            if num < 2:
                return False
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    return False
            return True
        except ValueError:
            raise ValueError("'math.is_prime' requires an integer base value")

    def math_degrees(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.degrees' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.degrees(num_base)
        except ValueError:
            raise ValueError("'math.degrees' requires a numeric base value")

    def math_radians(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.radians' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.radians(num_base)
        except ValueError:
            raise ValueError("'math.radians' requires a numeric base value")

    # --- Additional String Functions ---
    def string_pad_left(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.pad_left' requires a string base value")
        if len(args) not in (1, 2):
            raise ValueError(
                "'string.pad_left' requires 1 or 2 arguments: (length, [char])"
            )
        length = self._convert(args[0], _get_python_type_name(args[0]), "int")
        pad_char = " "
        if len(args) == 2:
            pad_char = self._convert(args[1], _get_python_type_name(args[1]), "string")
            if len(pad_char) != 1:
                raise ValueError("Padding character must be a single character")
        return base_value.rjust(length, pad_char)

    def string_pad_right(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.pad_right' requires a string base value")
        if len(args) not in (1, 2):
            raise ValueError(
                "'string.pad_right' requires 1 or 2 arguments: (length, [char])"
            )
        length = self._convert(args[0], _get_python_type_name(args[0]), "int")
        pad_char = " "
        if len(args) == 2:
            pad_char = self._convert(args[1], _get_python_type_name(args[1]), "string")
            if len(pad_char) != 1:
                raise ValueError("Padding character must be a single character")
        return base_value.ljust(length, pad_char)

    def string_capitalize(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.capitalize' requires a string base value")
        if len(args) != 0:
            raise ValueError("'string.capitalize' takes no arguments")
        return base_value.capitalize()

    def string_title(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.title' requires a string base value")
        if len(args) != 0:
            raise ValueError("'string.title' takes no arguments")
        return base_value.title()

    # --- Additional Random Functions ---
    def random_sample(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'random.sample' requires a list base value")
        if len(args) != 1:
            raise ValueError("'random.sample' requires exactly one argument: (size)")
        size = self._convert(args[0], _get_python_type_name(args[0]), "int")
        if size < 0 or size > len(base_value):
            raise ValueError("Sample size must be between 0 and list length")
        return random.sample(base_value, size)

    # --- Additional Type Functions ---
    def type_to_string(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'type.to_string' takes no arguments")
        return str(base_value)

    def type_to_bool(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'type.to_bool' takes no arguments")
        if base_type == "string":
            lower_val = base_value.lower()
            if lower_val in ("true", "1", "yes"):
                return True
            if lower_val in ("false", "0", "no"):
                return False
            raise ValueError("Cannot convert string to boolean")
        return bool(base_value)

    # --- Additional List Functions ---
    def list_reverse(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.reverse' requires a list base value")
        if len(args) != 0:
            raise ValueError("'list.reverse' takes no arguments")
        return base_value[::-1]

    def list_map(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.map' requires a list base value")
        if len(args) != 1:
            raise ValueError("'list.map' requires exactly one argument: (function)")
        if not callable(args[0]):
            raise ValueError("'list.map' argument must be a function")
        return list(map(args[0], base_value))

    def list_filter(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.filter' requires a list base value")
        if len(args) != 1:
            raise ValueError("'list.filter' requires exactly one argument: (function)")
        if not callable(args[0]):
            raise ValueError("'list.filter' argument must be a function")
        return list(filter(args[0], base_value))

    def list_reduce(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.reduce' requires a list base value")
        if len(args) not in (1, 2):
            raise ValueError(
                "'list.reduce' requires 1 or 2 arguments: (function, [initial])"
            )
        if not callable(args[0]):
            raise ValueError("'list.reduce' first argument must be a function")
        from functools import reduce

        if len(args) == 2:
            return reduce(args[0], base_value, args[1])
        return reduce(args[0], base_value)

    def list_find(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.find' requires a list base value")
        if len(args) != 1:
            raise ValueError("'list.find' requires exactly one argument: (value)")
        try:
            return base_value.index(args[0]) + 1
        except ValueError:
            return -1

    def list_count(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.count' requires a list base value")
        if len(args) != 1:
            raise ValueError("'list.count' requires exactly one argument: (value)")
        return base_value.count(args[0])

    def list_avg(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.avg' requires a list base value")
        if len(args) != 0:
            raise ValueError("'list.avg' takes no arguments")
        if not base_value:
            raise ValueError("Cannot calculate average of empty list")
        if not all(isinstance(x, (int, float)) for x in base_value):
            raise ValueError("'list.avg' requires all elements to be numeric")
        return sum(base_value) / len(base_value)

    def list_zip(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.zip' requires a list base value")
        if len(args) != 1:
            raise ValueError("'list.zip' requires exactly one argument: (list)")
        if not isinstance(args[0], list):
            raise ValueError("'list.zip' argument must be a list")
        return list(zip(base_value, args[0]))

    def list_enumerate(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.enumerate' requires a list base value")
        if len(args) > 1:
            raise ValueError("'list.enumerate' takes at most 1 argument: (start)")
        start = 0
        if len(args) == 1:
            start = self._convert(args[0], _get_python_type_name(args[0]), "int")
        return list(enumerate(base_value, start))

    def list_slice(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.slice' requires a list base value")
        if len(args) not in (1, 2):
            raise ValueError("'list.slice' requires 1 or 2 arguments: (start, [end])")
        start = self._convert(args[0], _get_python_type_name(args[0]), "int")
        if len(args) == 2:
            end = self._convert(args[1], _get_python_type_name(args[1]), "int")
            return base_value[start:end]
        return base_value[start:]

    def list_concat(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.concat' requires a list base value")
        if len(args) != 1:
            raise ValueError("'list.concat' requires exactly one argument: (list)")
        if not isinstance(args[0], list):
            raise ValueError("'list.concat' argument must be a list")
        return base_value + args[0]

    def list_unique(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.unique' requires a list base value")
        if len(args) != 0:
            raise ValueError("'list.unique' takes no arguments")
        return list(dict.fromkeys(base_value))

    # --- Time Functions ---
    def time_now(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'time.now' takes no arguments")
        import time

        return time.time()

    def time_sleep(self, base_value, base_type, args):
        if len(args) != 1:
            raise ValueError("'time.sleep' requires exactly one argument: (seconds)")
        import time

        seconds = self._convert(args[0], _get_python_type_name(args[0]), "float")
        if seconds < 0:
            raise ValueError("Sleep time cannot be negative")
        time.sleep(seconds)
        return None

    def time_format(self, base_value, base_type, args):
        if len(args) != 1:
            raise ValueError("'time.format' requires exactly one argument: (format)")
        import time

        if base_type not in ("int", "float"):
            raise ValueError("'time.format' requires a numeric timestamp base value")
        format_str = self._convert(args[0], _get_python_type_name(args[0]), "string")
        return time.strftime(format_str, time.localtime(base_value))

    def time_parse(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'time.parse' requires a string base value")
        if len(args) != 1:
            raise ValueError("'time.parse' requires exactly one argument: (format)")
        import time

        format_str = self._convert(args[0], _get_python_type_name(args[0]), "string")
        try:
            return time.mktime(time.strptime(base_value, format_str))
        except ValueError as e:
            raise ValueError(f"Time parsing error: {e}")

    # --- Additional File Functions ---
    def file_exists(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'file.exists' requires a string base value (filename)")
        if len(args) != 0:
            raise ValueError("'file.exists' takes no arguments")
        import os

        return os.path.exists(base_value)

    def file_delete(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'file.delete' requires a string base value (filename)")
        if len(args) != 0:
            raise ValueError("'file.delete' takes no arguments")
        import os

        try:
            os.remove(base_value)
            return True
        except OSError as e:
            raise ValueError(f"Error deleting file: {e}")

    def file_rename(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError(
                "'file.rename' requires a string base value (old filename)"
            )
        if len(args) != 1:
            raise ValueError(
                "'file.rename' requires exactly one argument: (new filename)"
            )
        import os

        new_name = self._convert(args[0], _get_python_type_name(args[0]), "string")
        try:
            os.rename(base_value, new_name)
            return True
        except OSError as e:
            raise ValueError(f"Error renaming file: {e}")

    def file_size(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'file.size' requires a string base value (filename)")
        if len(args) != 0:
            raise ValueError("'file.size' takes no arguments")
        import os

        try:
            return os.path.getsize(base_value)
        except OSError as e:
            raise ValueError(f"Error getting file size: {e}")

    # --- Additional System Functions ---
    def system_env(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError(
                "'system.env' requires a string base value (variable name)"
            )
        if len(args) != 0:
            raise ValueError("'system.env' takes no arguments")
        import os

        return os.getenv(base_value)

    def system_exec(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'system.exec' requires a string base value (command)")
        if len(args) != 0:
            raise ValueError("'system.exec' takes no arguments")
        import subprocess

        try:
            result = subprocess.run(
                base_value, shell=True, text=True, capture_output=True
            )
            return result.stdout.strip()
        except subprocess.SubprocessError as e:
            raise ValueError(f"Error executing command: {e}")

    # --- String Functions ---
    # NEW SIGNATURE: (self, base_value, base_type, args: List[Any])

    def string_contains(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.contains' requires a string base value")
        if len(args) != 1:
            raise ValueError("'string.contains' requires one argument: (substring)")
        substring = self._convert(args[0], _get_python_type_name(args[0]), "string")
        return substring in base_value

    def string_starts_with(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.starts_with' requires a string base value")
        if len(args) != 1:
            raise ValueError("'string.starts_with' requires one argument: (prefix)")
        prefix = self._convert(args[0], _get_python_type_name(args[0]), "string")
        return base_value.startswith(prefix)

    def string_ends_with(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.ends_with' requires a string base value")
        if len(args) != 1:
            raise ValueError("'string.ends_with' requires one argument: (suffix)")
        suffix = self._convert(args[0], _get_python_type_name(args[0]), "string")
        return base_value.endswith(suffix)

    def string_repeat(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.repeat' requires a string base value")
        if len(args) != 1:
            raise ValueError("'string.repeat' requires one argument: (count)")
        count = self._convert(args[0], _get_python_type_name(args[0]), "int")
        return base_value * count

    def string_reverse(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.reverse' requires a string base value")
        if len(args) != 0:
            raise ValueError("'string.reverse' takes no arguments")
        return base_value[::-1]

    def string_substr(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.substr' requires a string base value")
        if len(args) not in (1, 2):
            raise ValueError(
                "'string.substr' requires 1 or 2 arguments: (start, [length])"
            )
        start = self._convert(args[0], _get_python_type_name(args[0]), "int")
        length = None
        if len(args) == 2:
            length = self._convert(args[1], _get_python_type_name(args[1]), "int")
        return base_value[start : start + length] if length else base_value[start:]

    # --- New Math Functions ---
    def math_sin(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.sin' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.sin(num_base)
        except ValueError:
            raise ValueError("'math.sin' requires a numeric base value")

    def math_cos(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.cos' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.cos(num_base)
        except ValueError:
            raise ValueError("'math.cos' requires a numeric base value")

    def math_tan(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.tan' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.tan(num_base)
        except ValueError:
            raise ValueError("'math.tan' requires a numeric base value")

    # --- New Random Functions ---
    def random_choice(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'random.choice' requires a list base value")
        if len(args) != 0:
            raise ValueError("'random.choice' takes no arguments")
        if not base_value:
            raise ValueError("Cannot choose from empty list")
        return random.choice(base_value)

    def random_shuffle(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'random.shuffle' requires a list base value")
        if len(args) != 0:
            raise ValueError("'random.shuffle' takes no arguments")
        shuffled = base_value.copy()
        random.shuffle(shuffled)
        return shuffled

    # --- New Type Conversion Functions ---
    def type_to_int(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'type.to_int' takes no arguments")
        try:
            return self._convert(base_value, base_type, "int")
        except ValueError as e:
            raise ValueError(f"Cannot convert to int: {e}")

    def type_to_float(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'type.to_float' takes no arguments")
        try:
            return self._convert(base_value, base_type, "float")
        except ValueError as e:
            raise ValueError(f"Cannot convert to float: {e}")

    # --- New List Functions ---
    def list_sort(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.sort' requires a list base value")
        if len(args) > 1:
            raise ValueError("'list.sort' takes at most 1 argument (reverse)")
        reverse = False
        if len(args) == 1:
            reverse = self._convert(args[0], _get_python_type_name(args[0]), "bool")
        sorted_list = sorted(base_value)
        if reverse:
            sorted_list.reverse()
        return sorted_list

    def list_sum(self, base_value, base_type, args):
        if base_type != "list":
            raise ValueError("'list.sum' requires a list base value")
        if len(args) != 0:
            raise ValueError("'list.sum' takes no arguments")
        if not all(isinstance(x, (int, float)) for x in base_value):
            raise ValueError("'list.sum' requires all elements to be numeric")
        return sum(base_value)

    # --- New File Functions ---
    def file_read(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'file.read' requires a string base value (filename)")
        if len(args) != 0:
            raise ValueError("'file.read' takes no arguments")
        try:
            with open(base_value, "r") as f:
                return f.read()
        except IOError as e:
            raise ValueError(f"Error reading file: {e}")

    def file_write(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'file.write' requires a string base value (filename)")
        if len(args) != 1:
            raise ValueError("'file.write' requires one argument: (content)")
        content = self._convert(args[0], _get_python_type_name(args[0]), "string")
        try:
            with open(base_value, "w") as f:
                f.write(content)
            return True
        except IOError as e:
            raise ValueError(f"Error writing file: {e}")

    # --- New System Functions ---
    def system_args(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'system.args' takes no arguments")
        import sys

        return sys.argv[1:]  # Skip script name

    def system_exit(self, base_value, base_type, args):
        if len(args) > 1:
            raise ValueError("'system.exit' takes at most 1 argument (exit code)")
        code = 0
        if len(args) == 1:
            code = self._convert(args[0], _get_python_type_name(args[0]), "int")
        import sys

        sys.exit(code)

    def string_upper(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'string.upper' takes no arguments")
        if base_type != "string":
            raise ValueError("'string.upper' requires a string base value")
        return base_value.upper()

    def string_lower(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'string.lower' takes no arguments")
        if base_type != "string":
            raise ValueError("'string.lower' requires a string base value")
        return base_value.lower()

    def string_len(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'string.len' takes no arguments")
        # Works on lists too! Check type.
        if base_type not in ("string", "list"):
            raise ValueError(
                f"'string.len' requires a string or list base value, got {base_type}"
            )
        return len(base_value)

    def string_trim(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'string.trim' takes no arguments")
        if base_type != "string":
            raise ValueError("'string.trim' requires a string base value")
        return base_value.strip()

    def string_replace(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.replace' requires a string base value")
        if len(args) != 2:
            raise ValueError(
                "'string.replace' requires exactly two arguments: (old_str, new_str)"
            )

        old_str = self._convert(args[0], _get_python_type_name(args[0]), "string")
        new_str = self._convert(args[1], _get_python_type_name(args[1]), "string")
        return base_value.replace(old_str, new_str)

    def string_split(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.split' requires a string base value")
        if len(args) != 1:
            raise ValueError(
                "'string.split' requires exactly one argument: (delimiter)"
            )
        delimiter = self._convert(args[0], _get_python_type_name(args[0]), "string")
        # Handle splitting by empty string if desired, or disallow?
        # Python's split() with empty string raises ValueError. Let's mimic.
        if not delimiter:
            raise ValueError("Cannot split string by empty delimiter")
        return base_value.split(delimiter)

    def string_join(self, base_value, base_type, args):
        if base_type != "string":
            raise ValueError("'string.join' requires a string base value")
        if len(args) != 1:
            raise ValueError(
                "'string.join' requires exactly one argument: (list to join)"
            )
        base_value = self._convert(
            base_value, _get_python_type_name(base_value), "string"
        )
        lst = self._convert(args[0], _get_python_type_name(args[0]), "list")
        return base_value.join(lst)

    # --- Math Functions ---
    # NEW SIGNATURE: (self, base_value, base_type, args: List[Any])

    def math_sqrt(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.sqrt' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
        except ValueError:
            raise ValueError("'math.sqrt' requires a numeric base value")
        if num_base < 0:
            raise ValueError("Cannot calculate square root of a negative number")
        return math.sqrt(num_base)

    def math_pow(self, base_value, base_type, args):
        if len(args) != 1:
            raise ValueError("'math.pow' requires exactly one argument: (exponent)")
        try:
            num_base = self._convert(base_value, base_type, "float")
        except ValueError:
            raise ValueError("'math.pow' requires a numeric base value")
        try:
            # Get exponent from args[0]
            num_exp = self._convert(args[0], _get_python_type_name(args[0]), "float")
        except ValueError:
            raise ValueError("'math.pow' requires a numeric exponent argument")
        return math.pow(num_base, num_exp)

    def math_abs(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.abs' takes no arguments")
        try:
            target_type = "int" if base_type == "int" else "float"
            num_base = self._convert(base_value, base_type, target_type)
            return abs(num_base)
        except ValueError:
            raise ValueError("'math.abs' requires a numeric base value")

    def math_floor(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.floor' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.floor(num_base)
        except ValueError:
            raise ValueError("'math.floor' requires a numeric base value")

    def math_round(self, base_value, base_type, args):
        if len(args) > 1:
            raise ValueError("'math.round' requires at most 1 argument")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return round(num_base, args[0])
        except ValueError:
            raise ValueError("'math.round' requires a numeric base value")

    def math_min(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.min' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "list")
            return min(num_base)
        except ValueError:
            raise ValueError("'math.min' requires a list base value")

    def math_max(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.max' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "list")
            return max(num_base)
        except ValueError:
            raise ValueError("'math.max' requires a list base value")

    def math_ceil(self, base_value, base_type, args):
        if len(args) != 0:
            raise ValueError("'math.ceil' takes no arguments")
        try:
            num_base = self._convert(base_value, base_type, "float")
            return math.ceil(num_base)
        except ValueError:
            raise ValueError("'math.ceil' requires a numeric base value")

    # --- Random Functions ---
    # NEW SIGNATURE: (self, base_value, base_type, args: List[Any])
    def math_rnd(self, base_value, base_type, args: list[int]):
        """
        Generates a random number based on the number of integer arguments provided.
        """
        num_args = len(args)
        myrandom = random
        if base_value:
            myrandom = random.Random()
            myrandom.seed(base_value)

        if num_args > 2:
            # More specific error message is often helpful
            raise ValueError(f"'random.rnd' takes 0, 1, or 2 arguments, got {num_args}")

        if base_type != "int" and base_type != "void":
            raise ValueError(
                f"'random.rnd' requires an integer or void base value, recieved {base_type}"
            )

        match num_args:
            case 0:
                # No arguments: Mimic random.random()
                return myrandom.random()  # Returns float [0.0, 1.0)
            case 1:
                # One argument: Mimic random.randrange(stop)
                stop = args[0]
                if not isinstance(stop, int):
                    raise ValueError("random.rnd(stop): 'stop' must be an integer")
                if stop <= 0:
                    # random.randrange(stop) requires stop to be positive
                    raise ValueError(
                        "random.rnd(stop): argument 'stop' must be positive"
                    )
                return myrandom.randrange(stop)  # Returns int [0, stop)
            case 2:
                # Two arguments: Mimic random.randint(start, stop)
                start = args[0]
                stop = args[1]
                if not isinstance(start, int) or not isinstance(stop, int):
                    raise ValueError(
                        "random.rnd(start, stop): arguments must be integers"
                    )
                if start > stop:
                    # random.randint(a, b) requires a <= b
                    raise ValueError(
                        "random.rnd(start, stop): 'start' cannot be greater than 'stop'"
                    )
                # Returns int [start, stop]
                return myrandom.randint(start, stop)

    # --- Type Checking Functions ---
    # NEW SIGNATURE: (self, base_value, base_type, args: List[Any])

    def _type_check(
        self, base_type, target_type, args
    ):  # Added args for signature match
        if len(args) != 0:
            raise ValueError("Type check functions take no arguments")
        return base_type == target_type

    def type_is_int(self, base_value, base_type, args):
        return self._check_type(base_value, base_type, "int", args)

    def type_is_float(self, base_value, base_type, args):
        return self._check_type(base_value, base_type, "float", args)

    def type_is_string(self, base_value, base_type, args):
        return self._check_type(base_value, base_type, "string", args)

    def type_is_list(self, base_value, base_type, args):
        return self._check_type(base_value, base_type, "list", args)

    def type_is_bool(self, base_value, base_type, args):
        return self._check_type(base_value, base_type, "bool", args)

    def _check_type(self, base_value, base_type, expected_type, args):
        if self._debug:
            print(
                f"  Checking if type of {base_value} is {expected_type}: {base_value} (type: {base_type})"
            )
        return self._type_check(base_type, expected_type, args)

    def type_get_type(self, base_value, base_type, args):
        if self._debug:
            print(f"  Getting type of {base_value}: {base_type} (type: {base_type})")

        if len(args) != 0:
            raise ValueError("'type.get' takes no arguments")
        return base_type


def _get_python_type_name(value):
    """Maps Python types to the language's type names."""
    if isinstance(value, bool):
        return "bool"  # Check bool before int
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "void"
    return "any"
