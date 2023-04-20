from sgfmill import sgf, boards #Sgfmill is a library used for reading Smart Game Format files.
from copy import deepcopy #Used to create a true copy of an object and not a memory reference.
import torch
import numpy as np

boardsize =19
parse_dict = {None:0, 'b':1, 'w':-1} #Dictionary used for parsing the board representation that sgfmill produces.

import pickle 

with open('p_dict.pickle', 'rb') as handle:
    
    probability_dict = pickle.load(handle) #Loads precalculated probabilities
    
#np.random.seed(0) #During testing a seed is set so random numbers are predictable.

def open_game(path): #Helper function that reads SGF files with sgfmill and catches any errors that might occur  
                     #during reading.
    try:             
        with open(path, "rb") as f: #Using with statement ensures file is closed after reading.
            
            game = sgf.Sgf_game.from_bytes(f.read()) #Reads file with sgfmill.
            
        return game
    
    except:
        #print('File is not in SGF format') #Used during testing to help with debugging.
        
        return None #In the case file cannot be read function will return None so that other functions can deal with
                    #the error appropriately.

def input_board_to_tensor(input_board): #Helper function that takes the internal sgfmill representation of the go 
                                        #board and produces a corresponding tensor representation.
        
    np_input_board = np.array(input_board) #Converts a list of lists to a numpy array.
    
    np_input_board= np.vectorize(parse_dict.get)(np_input_board) #Parses the sgfmill representation of the go board.
    
    return torch.flip(torch.rot90(torch.from_numpy(np_input_board),-2),[1]) #Converts the numpy array to a torch 
                                                                            #tensor and flips and rotates the output
                                                                            #because of the strange internal 
                                                                            #formatting of the sgfmill board.
            

def process_playout_function(input_board, target): #Converts a sgfmill board and a set of coordinates into an input
                                                   #and output pair that can be used by the neural network.
        
    colour, (row, col) = target #Unpacking the data in the tuple target.
    
    row = 18-row #Because of the strange internal workings of sgfmill row coordinatws are stored inverted and 
                 #this must be reverted.
        
    input_tensor = input_board_to_tensor(input_board) 
    
    target = torch.zeros(boardsize,boardsize, dtype=int) #Creates a tensor full of zeros which will be the target.
    
    target[row, col] = 1 #Inserts a one in the position of the target move.
    
    assert input_tensor[row, col] == 0 #Makes sure that target is not an illegal move.
    
    return input_tensor, target, parse_dict[colour] #Returns the input and output pair alongside whose turn it is.


from random import choices #Built-in that allows for random weighted choices.

def playout_game(path): #Function that plays out a go game in an SGF format up to a certain point
                                            #and returns the last state as input and the next state as a target.
    game = open_game(path)  
    
    if not game: #Checks if opened SGF file is valid. 
        
            return None #None passed onto next function so that it is not added to dataset. 
        
    go_board = boards.Board(boardsize) #Creates a sgfmill board object to playout the game.
    game_len = len(game.get_main_sequence()) #Length of game.
    
    stop_idx = game_len -1 #Subtract one to avoid getting the last move in the sequence which by definition is 
                           #not followed by any other move.
        
    if stop_idx < 80: #If game is too short then it is not useful for training and is discarded.
        return None
    
    stop_idx = choices(list(probability_dict.keys())[:stop_idx], list(probability_dict.values())[:stop_idx])[0]
    #Uses the game length distribution.
    #Choosing random game lengths will mean that each dataloader that is created will have diffrent data and will
    #aid the neural network in generalizing.
    
    for moveidx, node in enumerate(game.get_main_sequence()): #Plays out game.
        
        if moveidx == stop_idx: #Checks if generated game length is reach to return the desired input and output pair.
            input_board = deepcopy(go_board.board) #Board attribute must be deepcopied in order to avoid getting
                                                   #a memory reference.
                
            target = node.get_move()
            
            if not target[0]:
                return None #Discards move if it a pass.
            
            colour, (row, col) = target
            
            try:
                go_board.play(row, col, colour) #Checks if move is valid.
                
            except: return None
            
            return (input_board, target, parse_dict[game.get_winner()]) #Returns the input to the neural network, 
                                                                        #the target and whose turn it is.
            
        if node.get_move()[1] == None: 
            continue
            
        try:
            
            colour, (row, col) = node.get_move() #Plays out game.
            go_board.play(row, col, colour) 
            
        except:
            #print('Invalid game')
            return None #If there is an error in the SGF it is discarded too.


from fastcore.parallel import parallel #Fastcore provides functions that are useful for threading and multiprocessing 
                                       #this has speed up the function open_games around 30x.
    
def open_games(paths_list): 
    
    opened_games=[]
    winners= []
    
    outputs = parallel(playout_game,paths_list, n_workers =31, chunksize= int(len(paths_list)/31)) 
    #Calls the function playout_game using 31 threads and splits the data into corresponding chunks.
                                                                                                    
    for output in outputs: 
        
        if not output: #Discards any games that have been deemed invalid. 
            continue
            
        input_board, target, winner = output #Unpacks the output.
        winners.append(winner)
        opened_games.append((input_board, target))
        
    return opened_games, winners 

def process_data(games):
    
    colours = []
    ground_truth=[]
    tensor_input=[]
    
    for game in games:

        input_tensor, target, colour = process_playout_function(*game) #Turns data into tensors so that can be used 
                                                                       #by the neural network.
        ground_truth.append(target)
        tensor_input.append(input_tensor)
        colours.append(colour)
        
    ground_truth = torch.stack(ground_truth) #Stacks the a list of 2d tensor into a 3d tensor.
    tensor_input = torch.stack(tensor_input)
    
    return tensor_input, ground_truth, colours
            