%% Test: String Operations

%% SEED
? 10

%% String concatenation
LET first: string = "Hello"
LET second: string = "World"
LET greeting: string = first + " " + second
SHOW(greeting)

%% String append
REAS greeting = greeting "!" [a]
SHOW(greeting)

%% String remove
REAS greeting = greeting "o" [r]
SHOW(greeting)

%% String insert
REAS greeting = greeting "o" [n] 5
SHOW(greeting)

%% String replace first occurrence
REAS greeting = greeting "l" [p] "L"
SHOW(greeting)

%% String replace all occurrences
REAS greeting = greeting "l" [P] "L"
SHOW(greeting)

%% String length
LET length: int = greeting [l]
SHOW(length)

%% String indexing
%% LET char: string = greeting [i] 1
%% SHOW(char)

%% Multiple string operations
LET text: string = "Hello Hello Hello"
SHOW(text)
REAS text = text "Hello" [p] "Hi"
SHOW(text)
REAS text = text "Hello" [P] "Hey"
SHOW(text)

%% Finding Items
LET x:int = text [f] "e"
SHOW(x)
REAS x = text [f] 100
SHOW(x)


%% Finding Items
LET s:string = text [?]
SHOW(s)
REAS s = text [?]
SHOW(s)

