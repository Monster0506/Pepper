%% Test: Conditional Statements

%% Simple IF
LET x: int = 10
LET y: int = 5

IF x > y DO
    SHOW("x is greater than y")
END;

%% IF-ELSE
IF x < 15 DO
    SHOW("x is less than 15")
ELSE DO
    SHOW("x is greater than or equal to 15")
END;

%% IF-ELIF-ELSE
LET score: int = 85
IF score > 90 DO
    SHOW("Grade: A")
ELIF score > 80 DO
    SHOW("Grade: B")
ELIF score > 70 DO
    SHOW("Grade: C")
ELSE DO
    SHOW("Grade: D")
END;

%% Nested IF statements
IF x > 0 DO
    IF y > 0 DO
        SHOW("Both x and y are positive")
    ELSE DO
        SHOW("Only x is positive")
    END;
END;

%% Boolean operators
LET a: bool = true
LET b: bool = false
IF a && b DO
    SHOW("Both are equal")
ELIF a &$$& b DO
    SHOW("Not equal")
ELSE DO
    SHOW("Both false")
END;
