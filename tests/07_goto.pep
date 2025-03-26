SHOW("Before jump")
GOTO 5;
SHOW("This will be skipped")
SHOW("This will be skipped too")
SHOW("After jump")

LET x: int = 5
REAS x = x 1 +
SHOW(x)
GOTO 8; x< 10