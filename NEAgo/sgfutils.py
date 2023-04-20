import numpy as np
from sgfmill import sgf, boards

boardsize =19
def open_game(path): 
    try:             
        with open(path, "rb") as f: 
            
            game = sgf.Sgf_game.from_bytes(f.read()) 
            
        return game
    
    except:
        print('Not in SGF format')
        return None 
    
parse_dict = {None:0, 'b':1, 'w':-1}

def input_board_to_numpy(input_board): 
    try:
        np_input_board = np.array(input_board) 

        np_input_board= np.vectorize(parse_dict.get)(np_input_board) 

        return np_input_board
    except:
        print('Could not process SGF')
        return None
            
def playout_game(path, move_idx): 
    game = open_game(path)  
    
    if not game: 
            return None 
    go_board = boards.Board(boardsize) 
    

    for idx, node in enumerate(game.get_main_sequence()): 
        if idx > move_idx:
            break
        try:
            colour, (row, col) = node.get_move() 
            go_board.play(row, col, colour) 
        except:
            continue 
    return input_board_to_numpy(go_board.board)