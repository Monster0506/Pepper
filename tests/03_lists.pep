%% Test: List Operations

%% List creation and basic operations
LET numbers: list = [1, 2, 3, 4, 5]
SHOW(numbers)

%% Append
REAS numbers = numbers 6 [a]
SHOW(numbers)

%% Remove
REAS numbers = numbers 3 [r]
SHOW(numbers)

%% Insert
REAS numbers = numbers 10 [n] 2
SHOW(numbers)

%% RePlace first occurrence
REAS numbers = numbers 4 [p] 40
SHOW(numbers)

%% RePlace all occurrences
REAS numbers = numbers 5 [P] 50
SHOW(numbers)

%% List length
LET length: int = numbers [l]
SHOW(length)

%% List indexing
LET item: int = numbers [i] 2
SHOW(item)

%% List from string
LET text: string = "Hello"
LET chars: list = text :> list
SHOW(chars)

%% Multiple operations
REAS numbers = numbers 100 [a]
REAS numbers = numbers 100 [P] 200
SHOW(numbers)
