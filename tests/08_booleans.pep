%% Testing boolean operators
LET x: int = 5
LET y: int = 5
LET z: int = 10
LET str1: string = "hello"
LET str2: string = "hello"
LET str3: string = "world"

%% Testing equality (&&)
SHOW("Testing equality (&&):")
SHOW(x && y)
SHOW(str1 && str2)

%% Testing inequality (&$$&)
SHOW("Testing inequality (&$$&):")
SHOW(x &$$& z)
SHOW(str1 &$$& str3)

%% Testing comparison operators
SHOW("Testing comparison operators:")
SHOW(x < z)
SHOW(z > x)
SHOW(x <= y)
SHOW(z >= x)

%% Testing logical operators (@$@, #$#, ~@)
SHOW("Testing logical operators:")
LET a: bool = true
LET b: bool = false

SHOW("a and b: a @$@ b:")
SHOW(a @$@ b)

SHOW("a or b: a #$# b:")
SHOW(a #$# b)

SHOW("not a: ~@ a:")
SHOW(~@ a)

SHOW("not b: ~@ b:")
SHOW(~@ b)

LET c: bool = x < z
LET d: bool = str1 && str2
LET e: bool = str1 &$$& str3
%% Testing complex conditions
SHOW("Testing complex conditions:")
SHOW(c @$@ d)
SHOW(~@ c #$# e)