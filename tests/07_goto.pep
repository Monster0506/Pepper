%% SIMPLE GOTO EXAMPLE
SHOW("Before jump")
GOTO 5;
SHOW("This will be skipped")
SHOW("This will be skipped too")
SHOW("After jump")

%% CONDITIONAL GOTO
LET x: int = 5
REAS x = x 1 +
SHOW(x)
GOTO 8; x < 10
SHOW("x < 10 : TRUE")
GOTO 8; x < 16


%% NESTED LOOPS 
LET a: int = 0
LET b: int = -1
REAS a = 1 a +
REAS b = 1 b +
SHOW("Inner:  " + b + "           Outer: " + a )
GOTO 20; b < 20
REAS b = -1
GOTO 19; a < 5
