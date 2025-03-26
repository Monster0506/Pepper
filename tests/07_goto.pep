%% SIMPLE GOTO EXAMPLE
SHOW("Before jump")
GOTO 6;
SHOW("This will be skipped")
SHOW("This will be skipped too")
SHOW("After jump")

%% CONDITIONAL GOTO
LET x: int = 5
REAS x = x 1 +
SHOW(x)
GOTO 10; x < 10
SHOW("x < 10 : TRUE")
GOTO 10; x < 16


%% NESTED LOOPS 
LET a: int = 0
LET b: int = -1
REAS a = 1 a +
REAS b = 1 b +
SHOW("Inner:  " + b + "           Outer: " + a )
GOTO 21; b < 20
REAS b = -1
GOTO 20; a < 5


%% GOTO with labels
LET m: int = 0
LBL a;
REAS m = m 1 +
SHOW(m)
GOTO a; m < 15
