LET options:list = ["scissors", "paper", "rock"]
LET p1choice:string = ""
LET p2choice:string = ""
LET result:int = 0
LET p1index:int = 0
LET p2index:int = 0
LET sub:int = 0
LET mod: int = 0
LET input:string = ""
LET again: string = ""


LBL start;
LBL p1picker;

REAS p1choice = INPT("Player 1: Enter one of ['rock', 'paper', 'scissors']")
REAS p1index = options [f] p1choice

GOTO p1picker; p1index && 0
REAS p2index = options [f] p2choice
REAS p2choice = options [?]

REAS sub = p1index p2index -
REAS mod = sub 3 %

SHOW("Player 2 Picked: " + p2choice)

REAS result = p1index p2index - 3 + 3 %
SHOW(result)

IF result && 0 DO 
    SHOW("It's a TIE!")
ELIF result && 1 DO
    SHOW("Player 2 Wins!")
ELSE DO 
    SHOW("Player 1 Wins!")
END

REAS input = INPT("Play again? [yes/no]")
REAS again =  input [i] 1
SHOW(again && "y")
GOTO start; again && "y"
