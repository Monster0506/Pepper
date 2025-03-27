LET options:list = ["scissors", "paper", "rock"]
LET p1choice:string = ""
LET p2choice:string = ""
LET p1index:int = 0
LET p2index:int = 0
LET sub:int = 0
LET mod: int = 0
LET input:string = ""


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


IF mod && 0 DO
    SHOW("It's a TIE!")
ELIF mod && 1 DO
    SHOW("Player 2 Wins!")
ELSE DO
    SHOW("Player 1 Wins!")
END

REAS input = INPT("Play again? [yes/no]")
GOTO start; input [i] 1 && "y"
