# Pepper Standard Library Reference

## String Library (`string`)
- `upper()` - Converts string to uppercase
- `lower()` - Converts string to lowercase
- `len()` - Returns length of string
- `trim()` - Removes leading/trailing whitespace
- `replace(old, new)` - Replaces all occurrences of 'old' with 'new'
- `split(delimiter)` - Splits string into list by delimiter
- `join(list)` - Joins list elements with string as delimiter
- `contains(substring)` - Checks if string contains substring
- `starts_with(prefix)` - Checks if string starts with prefix
- `ends_with(suffix)` - Checks if string ends with suffix
- `repeat(count)` - Repeats string count times
- `reverse()` - Reverses the string
- `substr(start, [length])` - Gets substring from start position
- `pad_left(length, char)` - Pads string left with char
- `pad_right(length, char)` - Pads string right with char
- `capitalize()` - Capitalizes first character
- `title()` - Converts string to title case

## Math Library (`math`)
- `sqrt()` - Square root
- `pow(exponent)` - Power function
- `abs()` - Absolute value
- `floor()` - Floor function
- `ceil()` - Ceiling function
- `round([decimals])` - Rounds number
- `min()` - Minimum value
- `max()` - Maximum value
- `sin()` - Sine function
- `cos()` - Cosine function
- `tan()` - Tangent function
- `asin()` - Arcsine function
- `acos()` - Arccosine function
- `atan()` - Arctangent function
- `log()` - Natural logarithm
- `log10()` - Base-10 logarithm
- `exp()` - Exponential function
- `pi()` - Returns π constant
- `e()` - Returns e constant
- `factorial()` - Factorial function
- `gcd()` - Greatest common divisor
- `lcm()` - Least common multiple
- `is_prime()` - Checks if number is prime
- `degrees()` - Converts radians to degrees
- `radians()` - Converts degrees to radians

## Random Library (`random`)
- `rnd()` - Random float between 0 and 1
- `rnd(stop)` - Random integer between 0 and stop-1
- `rnd(start, stop)` - Random integer between start and stop
- `choice()` - Random element from list
- `shuffle()` - Shuffles list randomly
- `sample(size)` - Random sample from list
- `seed(value)` - Sets random seed

## Type Library (`type`)
- `is_int()` - Checks if value is integer
- `is_float()` - Checks if value is float
- `is_string()` - Checks if value is string
- `is_list()` - Checks if value is list
- `is_bool()` - Checks if value is boolean
- `get()` - Gets type of value
- `to_int()` - Converts value to integer
- `to_float()` - Converts value to float
- `to_string()` - Converts value to string
- `to_bool()` - Converts value to boolean

## List Library (`list`)
- `sort([reverse])` - Sorts list
- `reverse()` - Reverses list
- `map(function)` - Maps function over list
- `filter(function)` - Filters list by function
- `reduce(function)` - Reduces list using function
- `find(value)` - Finds value in list
- `count(value)` - Counts occurrences in list
- `sum()` - Sums numeric list
- `avg()` - Averages numeric list
- `zip(list)` - Zips with another list
- `enumerate()` - Enumerates list with indices
- `slice(start, end)` - Gets list slice
- `concat(list)` - Concatenates lists
- `unique()` - Returns unique elements

## Time Library (`time`)
- `now()` - Current timestamp
- `sleep(seconds)` - Sleeps for seconds
- `format(format)` - Formats time string
- `parse(string)` - Parses time string

## File Library (`file`)
- `read()` - Reads file contents
- `write(content)` - Writes to file
- `exists()` - Checks if file exists
- `delete()` - Deletes file
- `rename(new_name)` - Renames file
- `size()` - Gets file size

## System Library (`system`)
- `args()` - Gets command line arguments
- `exit([code])` - Exits program
- `env(name)` - Gets environment variable
- `exec(command)` - Executes system command

## Usage Examples

```pepper
%% String operations
LET str: string = "Hello, World!"
LET upper: string = upper FROM string str _
LET contains: bool = contains FROM string str ("World")

%% Math operations
LET x: float = 3.14
LET sin_x: float = sin FROM math x _
LET pow_x: float = pow FROM math x (2)

%% Random operations
LET rand: float = rnd FROM random _ _
LET dice: int = rnd FROM random _ (1, 6)

%% List operations
LET nums: list = [1, 2, 3, 4, 5]
LET sorted: list = sort FROM list nums _
LET sum: int = sum FROM list nums _

%% File operations
LET content: string = read FROM file "test.txt" _
LET success: bool = write FROM file "output.txt" ("Hello")