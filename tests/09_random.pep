%% Test: Random Operations

%% SEED
? 10

%% Test random float between 0 and 1
LET random_float: float = ?
SHOW("Random float between 0 and 1: " + random_float)

%% Test random choice from list
LET fruits: list = ["apple", "banana", "orange", "grape", "mango"]
LET random_fruit: string = fruits [?]
SHOW("Random fruit: " + random_fruit)

%% Multiple random choices from same list
SHOW("Three random fruits:")
LET fruit1: string = fruits [?]
LET fruit2: string = fruits [?]
LET fruit3: string = fruits [?]
SHOW(fruit1)
SHOW(fruit2)
SHOW(fruit3)

%% Test random choice from string
LET word: string = "HELLO"
LET random_char: string = word [?]
SHOW("Random character from " + word + ": " + random_char)

%% Test random in a loop
LET rand: float = ?
SHOW("Five random numbers between 0 and 1:")
FOR i FROM 1 TO 5 DO
    REAS rand = ?
    SHOW(rand)
LOOP_END;

%% Test random integers in ranges
LET dice: int = ? 6 1 * +
SHOW("Random dice roll (1-6): " + dice)

%% Random number between 10 and 20
LET min: int = 10
LET max: int = 20
LET dif: int = max min -
LET range_num: int = ? max min - * min +
SHOW("Random number between " + min + " and " + max + ": " + range_num)

%% Multiple dice rolls
SHOW("Rolling five dice:")
FOR i FROM 1 TO 5 DO
    REAS dice = ? 6 * 1 +
    SHOW("Roll " + i + ": " + dice)
LOOP_END;
