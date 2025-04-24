LET options:list = ["rock", "paper", "scissors"]
LET p1choice:string = "paper"
LET p2choice:string = "rock"
LET result:int = 0
LET p1index:int = 0
LET p2index:int = 0
LET sub:int = 0
LET mod: int = 0
LET input:string = "no"
LET again: string = "yes"

LBL start;
LBL p1picker;

REAS p1choice = INPT("Player 1: Enter one of ['rock', 'paper', 'scissors']")
REAS p1index = options [f] p1choice
GOTO p1picker; p1index && 0

REAS p2choice = options [?]
REAS p2index = options [f] p2choice

SHOW("Player 2 Picked: " + p2choice)

REAS result = p1index p2index - 3 + 3 %

IF result && 0 DO 
    SHOW("It's a TIE!")
ELIF result && 1 DO
    SHOW("Player 1 Wins!")
ELSE DO 
    SHOW("Player 2 Wins!")
END

REAS input = INPT("Play again? [yes/no]")
REAS again =  input [i] 1
GOTO start; again && "y"
