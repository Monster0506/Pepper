? 10
LET name:string = "world"

LET upper_name:string = upper FROM string name _
SHOW(upper_name) %% Output: WORLD

SHOW(lower FROM string "TESTING 123" _) %% Output: testing 123

LET message:string = "Hello There"
LET length:int = len FROM string message _
SHOW(length) %% Output: 11

SHOW(sqrt FROM math 16.0 _) %% Output: 4.0
SHOW(sqrt FROM math 25 _)   %% Output: 5.0
LET x:int = 100
SHOW(sqrt FROM math x _)     %% Output: 10.0

LET three:int = 3
SHOW(pow FROM math 2 (three)) %% Output: 8.0
SHOW(pow FROM math 10 (2))    %% Output: 100.0

LET word:string = "mississippi"
LET replaced:string = replace FROM string word ("iss", "a") %% Args are now comma-separated
SHOW(replaced) %% Output: massassappa

LET csv:string = "apple,banana,cherry"
LET items:list = split FROM string csv (",") %% Argument directly in parentheses
SHOW(items) %% Output: ['apple', 'banana', 'cherry']


SHOW(is_string FROM type "hello" _)  %% Output: true
SHOW(is_int FROM type 5.0 _)        %% Output: false
SHOW(is_int FROM type 5 _)          %% Output: true
SHOW(get FROM type [true] _)        %% Output: list (renamed from get_type)

LET rand_num:float = rnd FROM random _ _ %% Base can be _, args is _
LET rand_num2:float = rnd FROM random 16 _ %% Base can be _, args is _
SHOW("Random float between 0 and 1: " + rand_num)
SHOW("Random scoped seed float between 0 and 1: " + rand_num2)
LET rand_int: int = rnd FROM random _ (10)
SHOW("Random int between 0 and 10: " + rand_int)
REAS rand_int = rnd FROM random _ (5, 15)
SHOW("Random int between 5 and 15: " + rand_int)
LET str:string = join FROM string ", " (["a", "b"])
SHOW(str) %% Output: ab
