from src.game.logic import GameLogic

class CLIGame:
    def __init__(self) -> None:
        self.logic = GameLogic()
    
    def printGame(self) -> None:
        game = self.logic
        board_str = str(game.board)
        print(board_str)

    def update(self) -> None:
        logic = self.logic
        match logic.state:
            case "start":
                _ = input("Press enter to start: ")
                logic.startGame()

                print("Game started.")
                self.printGame()

            case "game":
                if not logic.board.canMove(1) and not logic.board.canMove(-1):
                    logic.endGame()
                    return

                if logic.game_state == "player1":
                    if not logic.board.canMove(1):
                        print("No place left for Player1.")
                    else:
                        print("Player1's turn.")

                        while True:
                            x = int(input("X index: "))
                            y = int(input("Y index: "))
                            if logic.board.checkValidMove(player=1, pos=(x, y)):
                                logic.board.move(player=1, pos=(x, y))
                                break
                            else:
                                print("Try another position.")
                
                if logic.game_state == "player2":
                    if not logic.board.canMove(-1):
                        print("No place left for Player2.")
                    else:
                        print("Player2's turn.")
                        while True:
                            x = int(input("X index: "))
                            y = int(input("Y index: "))
                            if logic.board.checkValidMove(player=-1, pos=(x, y)):
                                logic.board.move(player=-1, pos=(x, y))
                                break
                            else:
                                print("Try another position.")
                
                self.printGame()
                logic.switchTurn()
            
            case "end":
                print("Game over.")
                print("Winner is: " + logic.winner_name)
                _ = input("Press enter to restart: ")
                logic.startGame()
                print("Game restarted.")
                self.printGame()

if __name__ == "__main__":
    manager = CLIGame()
    while True:
        manager.update()