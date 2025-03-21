%% Example usage of the interpreter with new data types

%% Basic variable declarations
LET x: int = 10
LET y: float = 3.14
LET name: string = "Pepper"
LET numbers: list = [1, 2, 3, 4, 5]
LET isvalid: bool = true

%% String manipulation
LET greeting: string = "Hello " + name
SHOW(greeting)

%% List operations
%% append
REAS numbers = numbers "a" [a]
SHOW(numbers)
%% remove
REAS numbers = numbers 2 [r]
SHOW(numbers)
%% iNsert
REAS numbers = numbers 3 [n] 2
SHOW(numbers)
%% length
LET count: int = numbers [l]
SHOW(count)
%% iNdex
LET item: int = numbers [i] 4
SHOW(item)
REAS item = numbers [i] 0
SHOW(item)
%% rePlace
REAS numbers = numbers "a" [p] "b"
SHOW(numbers)

%% Boolean expressions
REAS x = x 1 +
LET iseven: bool =  x 2 % && 0
SHOW("is even: " + x+ ": "+iseven)

%% Type conversions
LET pistr: string = y :> string
SHOW(pistr)
LET countfloat: float = count :> float
SHOW(countfloat)
REAS x = 10
REAS y = 3.14
LET z: int = 7

%% Conditionals
IF x > y DO
    SHOW("x>y")
    IF z > x DO
        SHOW("z>x")
    ELSE
        SHOW("z<=x " )
    END
ELIF x && y DO
    SHOW("x&&y")
ELSE
    SHOW("else")
END

LET a: int = 2
LET b: int = 2

%% not equal
IF a &$$& b DO
    SHOW(a)
END

SHOW(greeting)


%% Loops

LET start: int = 1
LET max: int = numbers [l] 1 +
LET m: int = 0
FOR i FROM start TO max DO
    REAS m = numbers [i] i
    SHOW(m)
LOOP_END


LET fancy: string = greeting + ", Hi"
SHOW(fancy)
%% string To List
LET ultrafancy: list = fancy :> list
SHOW(ultrafancy)
%% string Index
REAS a = fancy [i] -5
SHOW(a)
%% string Remove
REAS fancy = fancy "H" [r] 
SHOW(fancy)
%% string Append
REAS fancy = fancy "A" [a]
SHOW(fancy)
%% string Length
LET count2: int = fancy [l]
SHOW(count2)
%% string iNsert
REAS fancy =  fancy "A" [n] 4
SHOW(fancy)
%% string rePlace
REAS fancy = fancy "A" [p] "B"
SHOW(fancy)

%% string rePlace All
LET text: string = "Hello Hello Hello"
REAS fancy = text
SHOW(fancy)
REAS fancy = fancy "Hello" [p] "Hi"
SHOW(fancy)
REAS fancy = fancy "Hello" [p] "Hi"
SHOW(fancy)
REAS fancy = fancy "Hi" [P] "Hey"
SHOW(fancy)

