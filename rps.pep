LET p1: string = 0
LET p2: string = 0
LET options: list = ["scissors","paper","rock"]

LET p1index: int = 0
LET p2index: int = 0

LBL p1setup;
SHOW("Player 1: Enter one of ['rock', 'paper', 'scissors']")
REAS p1 = INPT("Player 1 CHOICE: ")
REAS p1index = options [f] p1
GOTO p1setup; p1index && 0


SHOW("Player 2: Enter one of ['rock', 'paper', 'scissors']")
REAS p2 = INPT("Player 2 CHOICE: ")
REAS p2index = options [f] p2
GOTO 15; p2index && 0


LET sub: int = p1index p2index -
LET len: int = options [l]
LET mod: int = sub len %

IF p1index && p2index DO
    SHOW("It's a TIE!")
ELIF mod && 1 DO
    SHOW("Player 2 Wins!")
ELSE DO
    SHOW("Player 1 Wins!")
END
