LET name:string = "world" 

LET upper_name:string = upper FROM name ()

SHOW(upper_name)

SHOW(lower FROM "TESTING 123" ()) %% Output: testing 123


LET message:string = "Hello There"
LET len:int = str_len FROM message ()
SHOW(len)


SHOW(sqrt FROM 16.0 ())
SHOW(sqrt FROM 25 ())
LET x:int = 100 
SHOW(sqrt FROM x ())

%% --- Examples using functions that DO take arguments ---

%% pow needs an exponent argument
LET three:int = 3
SHOW(pow FROM 2 (three))
SHOW(pow FROM 10 (2))

%% replace needs a list argument [old, new]
LET word:string = "mississippi"
LET replaced:string = replace FROM word (["iss", "a"])
SHOW(replaced)

%% split needs a delimiter string argument
LET csv:string = "apple,banana,cherry"
LET items:list = split FROM csv (",")
SHOW(items)


%% --- Type checking example ---
SHOW(is_string FROM "hello" ())
SHOW(is_int FROM 5.0 ())
SHOW(is_int FROM 5 ())
SHOW(get_type FROM [true] ())