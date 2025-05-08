determineWinner::(p1:string, p2:string) ->
  LET options: list = ["rock", "paper", "scissors"]
  LET i1: int = options [f] p1
  LET i2: int = options [f] p2
  RETURN i1 i2 - 3 + 3 %
<- int

LET p1choice: string ="paper"
LET p2choice: string ="paper"
LET result: int = 0
LET p1index: int = 0

LET input: string = "no"
LET again: string = "y"

LET options: list = ["rock", "paper", "scissors"]
LBL start;

LBL p1picker;
REAS p1choice = INPT("Player 1: Enter one of ['rock', 'paper', 'scissors']")
REAS p1index = options [f] p1choice
GOTO p1picker; p1index && 0

REAS p2choice = options [?]

SHOW("Player 2 Picked: "+ p2choice)



REAS result = (p1choice, p2choice) |> determineWinner
IF result && 0 DO
  SHOW("Tie")
ELIF result && 1 DO
  SHOW("p1 wins")
ELSE DO
  SHOW("p2 wins")
END


REAS input = INPT("Play again? [yes/no]")
REAS again= lower FROM string input [i] 1 _
GOTO start; again && "y"

