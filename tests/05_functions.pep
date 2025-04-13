%% Test: Function Declaration and Usage
IMPORT my_func FROM my_module

%% Basic function with no parameters
printHello::()->
    SHOW("Hello from function!")
<- void

%% Call the function
_ |> printHello


%% Function with parameters
greet::(name: string)->
    SHOW("Hello, " + name + "!")
<-void

%% Call with parameter
("Alice") |> greet
("Bob") |> greet

%% Function with multiple parameters and return value
add::(x:int, y:int)->
    LET result: int = x y +
    SHOW("Adding " + x + " and " + y)
    RETURN result
<-int

%% Call and use return value
LET sum: int = (5, 3) |> add
SHOW("Sum is: " + sum)

%% Test with mixed types
LET message: string = (3.14, "pi") |> calculate
SHOW(message)

%% Function with different types
calculate::(num:float, text:string)->
    LET result: string = num :> string + " " + text
    RETURN result
<-string

SHOW((5, "result") |> calculate)
%% Test imported function
LET a: int = 5
SHOW((a) |> my_func)
