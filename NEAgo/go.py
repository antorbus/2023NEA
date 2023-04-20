from scipy.signal import convolve2d
import numpy as np
import pygame
from itertools import product
import time 
import torch 
import pickle
import sys

from sgfutils import playout_game
from mergesort import mergesort
from model import GoModel

class Go():
    def __init__(self):
        
        self.size = 19 #Stores the size of the go board.
        
        self.board_numerical = np.full((self.size, self.size), 0) #2d array full of 0s representing go board.
        
        self.pos_map = np.arange(self.size**2).reshape(self.size,self.size) #2d array with positions from the 
                                                                            #flattened array.
            
        self.kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]]) #Kernel used in the convolutional operation that determines liberties 
                                            #on the board.
        
        self.update_liberty_map() #Creates liberty map.
        
        self.colour_num = {1:'Black', -1:'White'} #Numbers to colour. 
        
        self.scores = {1:0,-1:0} #Keeps track of scores.
        
        self.turn = 1 #Stores current turn.
        
        self.passes = 0 #Stores number of consecutive passes.
        
        self.past_boards = [None, None] #Queue that stores previous two boards to check moves prohibited by ko.
        
        
        self.diagonal_recurision = False #Changes how the recursive algorithm traverses the board 
                                         #is set to true in the endgame to remove dead stones 
        
        self.alive_pos_groups = {1:[],-1:[]} #Stores position of groups which are alive in the end game.
        
        self.model = None #AI models is stored here
        
    
    def __call__(self): #Overriding the call dunder method so that the main menu opens when an instance of the 
                        #Go class is called.
            
        self.main_menu()
        
    def use_sgf_game(self): #Uses the stored SGF game alongside other auxiliary data to load in a game.
        try:
            with open('sgf_game/data.pickle', 'rb') as handle: #Auxiliary data must be stored in the file 
                                                               #data.pickle.
                    
                score_turn = pickle.load(handle)
                    
            self.scores, self.turn, move_idx = score_turn #Scores and turn extracted and updated.
                
            game = playout_game('sgf_game/game.sgf', move_idx) #Plays out game.
        
            if type(game) == np.ndarray: #Prevents the game from crashing in the future if incorrect file loads.
            
                self.board_numerical = game 
                return True
                
        except:
            
            print('Loaded incorrectly') 
        return False
       
        
        
    def instantiate_models(self): #Model instantiated and loaded.
        try:
            self.model = GoModel() 
     
            self.model.load_state_dict(torch.load('models/model.ckpt', map_location=torch.device('cpu'))) 
                #Map location is set to CPU because this code is supposed to run on the proposed user.
            return True
        except:
            return False
            
    def model_evaluation(self):
        
        
        colour_map = torch.full((self.size, self.size), self.turn) #Tensor populated with the value of 
                                                                   #turn so model knows whose turn it is.
        
        board = torch.from_numpy(self.board_numerical) 
        
        #Splitting the board state into two board states with black and white positions.
        net_input_black = board == 1
        net_input_white = board == -1
        
        #Combining all of the inputs into a 4d tensor to feed as input into the model.
        inputs = torch.stack((net_input_black.float(), net_input_white.float(), colour_map)).unsqueeze(0)
        
        move_evaluation, winning = self.model(inputs) #Model predictions       
        
        #Removing any illegal predictions.
        legal_board = self.get_torch_legal_moves() 
        move_evaluation = torch.where(legal_board, move_evaluation.squeeze(0), 0)

        #Selecting what the AI thinks are the top 10% moves.
        split_val = torch.quantile(move_evaluation, 0.9) 
        move_evaluation = torch.where(move_evaluation > split_val, move_evaluation, 0).squeeze(0)
        
        #Moves scaled as values from 0 to 1.
        move_evaluation = move_evaluation/move_evaluation.max()
        
        #Selecting the player the AI thinks is winning.
        winning =  self.colour_num[int(torch.sign(winning).item())]
        
        #Detach is required to convert tensor to numpy array since numpy does not store gradients.
        return move_evaluation.detach().numpy(), winning
    
    
    def update_liberty_map(self): #Updates the liberty map by applying a convolution operation to the board.
        
        self.liberty_map = 4 - convolve2d(np.abs(self.board_numerical), self.kernel, mode="same", fillvalue= 1)
    
    
    def pos_rowcol(self, x): #Takes either a tuple with coordinates or the flattened index of the coordinates 
        #(polymorphism) and converts them.
        
        try:
            
            if type(x) == tuple: #Row column to pos.
                
                r, c = x #Unpacking row and column.
                
                return r*self.size + c
            
            else : #Pos to row column.
                
                return (x //self.size, x %self.size)
            
        except: 
            
            raise ValueError #Raises value error if incorrect data type passed into function.
    
    
    def adjacency_recursive(self, pos, colour): #Wrapper method that calls adjacency recursive.
        
        self._adjacency_list = {} #Creates an adjacency list and stores it as a class attribute.
                                  #This minimizes the size of the stackframe and makes the function take less 
                                  #memory as opposed to passing it as a parameter. 
        
        self._board_bool = self.board_numerical == colour #Creates a board with 1s where the players are.
                                                          #This facilitates checking what positions can be 
                                                          #traversed.
        
        self._board_bool = self._board_bool.reshape(self.size,self.size)
        
        self._adjacency_recursive(pos) #Calls adjacency recursive with the starting position.
        
        return mergesort(self._adjacency_list) #Sorts the output.
       
        
    def _adjacency_recursive_helper(self, pos, row, col): #Helper function used in the main adjacency recursive
                                                          #function that stores the new positions of the 
                                                          #traversed array. 
                                                          #Coded in a separate function to improve readability.
            
        pos_new = self.pos_rowcol((row, col))
        
        self._adjacency_list[pos].append(pos_new)
        
        self._adjacency_recursive(pos_new) #Recursive statement.
     
    
    def _adjacency_recursive(self, pos): #Function used to find all of the positions of stones that form part of
                                         #a group.
            
        if pos not in self._adjacency_list: #Recursive base case.
            
            row, col = self.pos_rowcol(pos)
            
            self._adjacency_list[pos] = [] #New group created in the adjacency list.
            
            #Finds all adjacent stones of the same colour and calls adjacency recursive on them too.
            for ax, edge, (delta_r, delta_c) in zip([col, col, row, row], [0, self.size-1, 0, self.size-1],[(0,-1), (0,1), (-1,0), (1,0)]):
                
                if ax != edge and self._board_bool[row+delta_r, col+delta_c]:
                    
                    self._adjacency_recursive_helper(pos, row+delta_r, col+delta_c)
            
            #Includes diagonal positions too.
            if self.diagonal_recurision:
                
                for (edge_r, edge_c), (delta_r, delta_c) in zip([(0,0),(0,self.size-1),(self.size-1, 0),(self.size-1, self.size-1)],[(-1,-1),(-1,1),(1,-1),(1,1)]):
                    
                    if row != edge_r and col != edge_c and self._board_bool[row+delta_r, col+delta_c]:
                        
                        self._adjacency_recursive_helper(pos, row+delta_r, col+delta_c)

                    
    def group_finder(self, colour): #Function that finds all of the groups (of a certain colour) on the board 
                                    #and their liberties.
        
        self.update_liberty_map() #Liberty map updated to newest state.
        
        indices = list(np.where((self.board_numerical ==colour).flatten())[0]) #All the positions of the stones
                                                                               #of that colour.
        
        adjacency_liberty = [] #Stores positions and tuples of groups.
        
        while indices: #Indices is used to keep track of positions which have not been visited.
            
            adjacency_list = self.adjacency_recursive(indices[0], colour) #Finds all of the coordinates of 
                                                                          #a group.
            
            index = list(adjacency_list.keys())
            
            #Stores the positions alongside the total number of liberties of that group.
            adjacency_liberty.append((adjacency_list, self.liberty_map.flatten()[index].sum())) 
            
            indices = [i for i in indices if i not in index] #Removes traversed positions.
            
        return adjacency_liberty
    
    def dead_stones(self, colour, update_score=False): #Function that finds and kills of the the opponents groups
                                                       #if they are surrounded.
        
        groups = self.group_finder(-1*colour) #Finds all of the oppenents groups.
        
        killed = False
        
        for group, liberties in groups:
            
            if liberties == 0: #If a group is surrounded.
                
                coords = np.array([[*self.pos_rowcol(x)] for x in group.keys()]).T #Gets all of the coordinates
                                                                                   #in row column format.
                
                self.board_numerical[coords[0], coords[1]]= 0 #Kills the group.
                
                if update_score:
                    
                    self.scores[colour] += len(group) #If this function is called when a move is made it will
                                                      #also update the player's score.
                    
                killed = True
                
        return killed #Also returns if it has killed any group.
                      #This is used to determine if certain moves are suicide.
                
    def to_list_board(self, board):
        
        return board.flatten().tolist() #Numpy 2d array to list.

    
    def ko(self): #Checks if the ko rule is broken by a certain move.
        
        assert len(self.past_boards) ==2 #Checks the stack has a constant size.
        
        if self.past_boards[0] == self.to_list_board(self.board_numerical): #If a position is repeated then
                                                                            #a true is returned.
            print('Ko')
            
            return True
        
        else:
            
            return False
    
    def suicide(self, row, col, colour): #Checks if a certain group has any liberties left after a move.
        
        adjacency_list = self.adjacency_recursive( self.pos_rowcol((row,col)), colour) #Finds positions of group.
        
        index = list(adjacency_list.keys())
        
        if not self.liberty_map.flatten()[index].sum(): #If the group has zero liberties then it is suicide.
            
            print('Suicide')
            
            return True
        
        return False
    
    def is_legal_move(self, row, col): #Checks if a particular move is legal.
        
        if not self.board_numerical[row][col] : #Checks board is not occupied at a certain spot.
            
            board = self.board_numerical.copy() #Stores a copy of the board to reset it after it has been altered.
            
            self.board_numerical[row][col] = self.turn #Makes move to check if it is legal.
            
            kill = self.dead_stones(self.turn) #Kills the oppenents stones.
            
            if not self.ko(): #Checks if ko is broken.
                
                if kill : #If it is not broken and a group is captured then move is legal.
                    
                    pass
                
                elif self.suicide(row, col, self.turn): #Checks if move would result in suicide.
                    
                    self.board_numerical=board #If it does board is reset and deemed invalid
                    
                    return None
                
                self.board_numerical=board #Board is reset.
                
                return True
            
            else: 
                
                self.board_numerical=board
            
    def get_legal_moves(self): #Finds all legal moves on the board.
        
        moves = np.where((self.board_numerical ==0)) #Finds all possible moves.
        
        legal_moves=[]
        
        for row, col in zip(*moves): #Iteratively checks them all to see if they are legal/
            
            if self.is_legal_move(row, col):
                
                legal_moves.append((row,col)) #Legal moves are stored in a list.
                
        return legal_moves
    
    def get_torch_legal_moves(self): #Calls get_legal_moves and stores the legal moves in a tensor.
                                     #This is done so any bad moves made by the AI can be filtered out.
        
        legal_board=torch.zeros((self.size,self.size)) #Board of legal moves.
        
        legal_moves = self.get_legal_moves()
        
        for row, col in (legal_moves):
            
            legal_board[row,col]=1 #Sets position from 1 to 0 if move is legal.
            
        return legal_board.bool() #Turns it into an array of booleans.
    
    def next_turn(self): #Changes turn.
        
        self.turn *= -1
        
    def move(self, row, col): #Makes move.
        
        if self.is_legal_move(row,col): #Checks if move is legal.
            
            self.board_numerical[row][col] = self.turn #If it is then the move is made.
            
            self.dead_stones(self.turn, True) #Opponents stones are removed from the board if any are killed.
                                              #Score is also updated.
            
            self.passes = 0 #Passes set to 0 since a move has been made.
            
            self.next_turn() #It is now the next player's turn.
            
            #Past boards stack is updated.
            self.past_boards.pop(0) #Pop.
            self.past_boards.append(self.to_list_board(self.board_numerical)) #Push.
            return True
        else:
            return False
    
    def pass_turn(self): #If a pass is made then pass counter incremented and it is now the opponent's turn.
        
        self.next_turn()
        
        self.passes += 1
        
    def determine_life(self, pos_group): #Checks if a group is alive or if it is not.
        
        board = np.zeros(self.size**2, dtype =int) #New board created.
        
        board[pos_group] =1 #Group added to board.
        
        board= board.reshape(self.size,self.size)
        
        not_group = np.stack(np.where(board == 0), axis =1).tolist() #All of the positions which are not part of
                                                                     #the group.
        visited = [] 
        
        stack = [not_group[0]]
        
        while stack: #Will try to traverse all of the positions which are not part of the group.
                     #If it is able to then group is not alive.
            
            current_node = stack.pop()
            
            if current_node in visited: #Checks if it has already visited that node.
                continue
                
            visited.append(current_node) #If it has not then it is added to visited nodes.
            
            row, col = current_node
            
            #Checks if adjacent positions can be traversed and adds them to the stack if they can.
            if col!=self.size-1 and not board[row][col+1]: stack.append([row,col+1])
                
            if row!=self.size-1 and not board[row+1][col]: stack.append([row+1,col])
                
            if col!= 0 and not board[row][col-1]: stack.append([row,col-1])
                
            if row!= 0 and not board[row-1][col]: stack.append([row-1,col])
                
        if len(visited) == len(not_group): #If all positions have been traversed then group is not alive.
            
            return False
        
        else: 
            return True
        
        raise Exception
    
    def endgame_removal(self, colour): #Removes dead stones when a game is finished.
        
        self.diagonal_recurision = True #Diagonal recurision enables the recursive algorithm the capability of
                                        #finding groups connected by diagonals.
                                        #Diagonal recursion was not necessary before since the algorithm did not
                                        #need to find true groups. 
        
        indices = list(np.where((self.board_numerical ==colour).flatten())[0])
        
        adjacency = []
        
        while indices: #Finds all true groups on the board.
            
            adjacency_list = self.adjacency_recursive(indices[0], colour) 
            
            index = list(adjacency_list.keys())
            
            adjacency.append(index)
            
            indices = [i for i in indices if i not in index]
            
        for pos_group in adjacency:
            
            life = self.determine_life(pos_group) #Determines if all of those groups are alive.
            
            if not life: 
                
                self.scores[colour] -= len(pos_group) #If group is dead then score is subtracted from player.
                
                self.board_numerical = self.board_numerical.flatten()
                
                self.board_numerical[pos_group] = 0 #Group removed from board.
                
                self.board_numerical.resize(self.size, self.size)
                
            else:
                
                self.alive_pos_groups[colour].append(pos_group) #Used for debugging.
    
    def territory_counter_traverse(self, colour, pos): #Counts the total area surrounded by each group.
        
        stack = [pos]
        
        visited = [] 
        
        encountered_opp = False #If the algorithm encounters a an opponent's piece then that all of the the 
                                #territory traversed does not count to the player's score
        
        while stack:
            
            current_node = stack.pop()
            
            row, col = current_node
            
            if self.board_numerical[row, col] == -1*colour: #Checks if an opponent's piece is encountered.
                
                encountered_opp = True
                
            if current_node in visited: continue
                
            visited.append(current_node)
            
            if col!=self.size -1 and self.board_numerical[row][col+1] != colour:
                
                stack.append([row,col+1])
                
            if row!=self.size-1 and self.board_numerical[row+1][col] != colour:
                
                stack.append([row+1,col])
                
            if col != 0 and  self.board_numerical[row][col-1] != colour:
                
                stack.append([row,col-1])
                
            if row != 0  and  self.board_numerical[row-1][col] != colour:
                
                stack.append([row-1,col])
                
        if not encountered_opp: #If no opponent has been found then the territory traversed does count
                                #towards the player's score.
            
            self.scores[colour] += len(visited)
            
        return visited
        
                
    def territory_counter(self, colour): #Applies territory_counter_traverse to all of the positions of the board.
        
        positions = np.stack(np.where(self.board_numerical == 0), axis =1).tolist()
        
        while positions:
            
            visited = self.territory_counter_traverse(colour, positions.pop())
            
            positions = [pos for pos in positions if pos not in visited]
    
    def add_komi(self):
        
        self.scores[-1] += 6.5 #Adds komi as compensation to white for playing second.
        
    def calculate_score(self):
        
        #Both white's and black's stones are removed.
        self.endgame_removal(1) 
        self.endgame_removal(-1)
        
        #Calculates scores for both players.
        self.territory_counter(1)
        self.territory_counter(-1)
        
        self.add_komi() #Adjusting for komi.
        
        if self.scores[1] > self.scores[-1]: #Returns who the winner is and by how much.
            
            return [1, self.scores[1]-self.scores[-1]]
        
        else: 
            
            return [-1, self.scores[-1]-self.scores[1]]
        
    def quit(self): 
        
        #Quits out of everything after a game has finished or the window is closed.
        pygame.display.quit()
        pygame.quit()
        sys.exit()
       
    def draw_text(self, text, color, surface, x, y): #Helper function that is used to render text .
        
        textobj = self.font.render(text, 1, color) #Instantiating text object.
        
        textrect = textobj.get_rect() #Getting the pygame rectangle object of the text.
        
        textrect.topleft = (x, y) #Using it to set the position of the text.
        
        surface.blit(textobj, textrect) #Representing the text on the board.
        
    def fill_option_screen(self, screen): #Function that is called when the option menu is opened.
        
        screen.fill(self.screen_fill) #Sets the colour of the screen.
        
        #Creates three buttons.
        button_1 = pygame.Rect(250, 230, 200, 50)
        button_2 = pygame.Rect(250, 390, 200, 50)
        button_3 = pygame.Rect(250, 310, 200, 50)
        
        pygame.draw.rect(screen, (203, 171, 11), button_1) #Draws the buttons on the screen.
        pygame.draw.rect(screen, (203, 171, 11), button_2)
        pygame.draw.rect(screen, (203, 171, 11), button_3)
        
        self.draw_text('Load SGF',self.white, screen, 270, 245) #Draws text on top of the buttons.
        self.draw_text('Load model', self.white, screen, 270, 325)
        self.draw_text('Back', self.white, screen, 270, 405)
        
        return button_1, button_2, button_3 #Returns buttons so that checking if they are clicked is easy.
        
    def options_screen(self, screen, clock):
        
        done= False 
              
        button_1, button_2, button_3 = self.fill_option_screen(screen) #Gets buttons.
        
        pygame.display.update() #Draws buttons.
        
        while not done:
            
            click = False 
            
            for event in pygame.event.get():
                
                if event.type == pygame.QUIT: self.quit() #If GUI is closed then program quits out of everything.
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: click = True #If there is a click then click is set to true.
                                    
            mx, my = pygame.mouse.get_pos() #Gets position of mouse.

            if button_1.collidepoint((mx, my)): #Checks if there is a click and button 1 is pressed.
                
                if click: 
                    
                    sgf_on = self.use_sgf_game() #Loads SGF.
                    
                    #Prints a text on a screen that shows the user if the SGF was loaded correctly.
                    if sgf_on: 
                        self.draw_text('SGF Loaded',self.black, screen, 460, 245)
                    else:
                        self.draw_text('Could not load SGF',self.black, screen, 460, 245)
                    
                    pygame.display.update()
                    
            #If button 2 is pressed then user goes back to the main menu.
            if button_2.collidepoint((mx, my)):
                if click: done = True
                    
                    
            #Checks if button 3 is pressed.       
            if button_3.collidepoint((mx, my)):
                
                if click: 
                    
                    loaded = self.instantiate_models() #Loads in the model.
                    
                    
                    #If model was loaded in correctly then it shows the user.
                    if loaded:
                        self.draw_text('Loaded model', self.black, screen, 460, 325)
                    else:
                        self.draw_text('Could not load model', self.black, screen, 460, 325)
                        
                    pygame.display.update() 
                    
            clock.tick(60)
            
    def fill_main_menu(self, screen):
        
        screen.fill(self.screen_fill)
        
        #Sets the font as a class attribute so it can be used in different parts of the program.
        self.font = pygame.font.SysFont('Corbel',35)
        
        #Play and Options buttons.
        button_1 = pygame.Rect(250, 230, 200, 50)
        button_2 = pygame.Rect(250, 310, 200, 50)
        
        #Drawing the buttons.
        pygame.draw.rect(screen, (203, 171, 11), button_1)
        pygame.draw.rect(screen, (203, 171, 11), button_2)
        
        self.draw_text('GO MAIN MENU', self.black, screen, 257, 100)
        self.draw_text('PLAY',self.white, screen, 270, 245)
        self.draw_text('OPTIONS', self.white, screen, 270, 325)
        
        pygame.display.update() #Showing them on the screen.
        
        return button_1, button_2
        
    def main_menu(self):
        
        pygame.init()
        
        self.black, self.white = (0, 0, 0), (255, 255, 255)
        self.window_dim = [735, 770] #Setting the window size.
        self.screen_fill = (199, 143, 30) #Sets the background colour for the screen. 
        screen, clock = pygame.display.set_mode(self.window_dim), pygame.time.Clock() #Sets the screen and clock.
        
        button_1, button_2 = self.fill_main_menu(screen) #Play and options buttons.
        
        done = False
        
        while not done:
            click = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        click = True


            mx, my = pygame.mouse.get_pos()
            
            if button_1.collidepoint((mx, my)):
                if click:
                    self.play( screen, clock)
                    done = True
                    
            if button_2.collidepoint((mx, my)):
                if click: 
                    self.options_screen(screen, clock)
                    self.fill_main_menu(screen)

            clock.tick(60)
    
    def play(self, screen, clock):
        
        done = False
        
        hoshi = list(product([3,9,15], repeat =2)) #Getting the coordinates of the 9 hoshi points.
        grey_b, grey_w, red =  (150,150,150),(200,200,200), (255,0,0)
        dim, margin, div, mul = 15, 23, 3, 45.91 #Setting some constants 
        
        greys=[None, grey_b, grey_w] #None is added to the front of the list so that choosing [1] and [-1] will 
                                     #return grey_b and grey_w.
        if self.model:
            moves, win = self.model_evaluation()  #If a model is loaded then the predictions are made.
            
        while not done: 
            
            for event in pygame.event.get():  
                
                if event.type == pygame.QUIT:  
                    
                    self.quit()
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    
                    pos = pygame.mouse.get_pos()
                    
                    #Turn a mouse postions into a row and a column.
                    col = pos[0] // (dim + margin) 
                    row = pos[1] // (dim + margin)
                    
                    if col < self.size and row < self.size:  #If click is inside of the board.
                        
                        moved = self.move(row, col) #Make move.
                        
                        if moved and self.model: #If move han been made and model is loaded then new evaluation
                                                 #is made.
                                
                                moves, win = self.model_evaluation()
                    
                    #If the pass button is clicked then pass turn is called.
                    elif margin+10 <= pos[0] <= margin+10+100 and 724 <= pos[1] <= 724+40:
                        
                        self.pass_turn()
            
            #Screen is refilled with yellow.
            screen.fill(self.screen_fill)
            
            #Drawing the pass button.
            pygame.draw.rect(screen,greys[self.turn],[margin,724,100,40])
            
            #Drawing whose turn it it, the number of captures and the pass button text.
            self.draw_text(f'{self.colour_num[self.turn]} turn', self.white, screen, margin+400,726)
            self.draw_text(f'Captures B {self.scores[1]} W {self.scores[-1]}', self.white,screen,margin+130,726)
            self.draw_text('Pass',self.white,screen, margin+10,729)
            
            
            #Draws the horizontal and vertical lines of the Go board.
            for row in range(19):
                for column in range(19):
                    pygame.draw.rect(screen, self.black, [margin ,(margin + dim) * row + margin ,dim*mul, dim/div])
                    pygame.draw.rect(screen, self.black, [(margin + dim) * row + margin , margin ,dim/div, dim*mul])
                    
            #Draws the 9 hoshi points on the board.
            for row, col in hoshi:
                pygame.draw.circle(screen,self.black,((margin + dim) * col + margin +2,(margin + dim)*row+margin+2),8)
            
            #Selects all of the black points and white points.
            black_pos = np.where(self.board_numerical == 1)
            white_pos = np.where(self.board_numerical == -1)
            
            #Draws all of the black stones on the board.
            for row, col in zip(*black_pos):
                pygame.draw.circle(screen, self.black,((margin + dim) * col + margin+2 ,(margin +dim)*row+margin+2),15)
            
            #Draws all of the white stones on the board.
            for row, col in zip(*white_pos):
                pygame.draw.circle(screen,self.white,((margin + dim) * col + margin+2 ,(margin +dim)*row+margin+2),15) 
                
            #If a model is loaded then it draws the points which it has predicted. 
            if self.model:
                
                preds = np.where(moves !=0 ) #Selects the coordinates of the predicted moves.
                
                self.draw_text(f'Eval: {win}',self.white, screen, 560,726) #Shows who the AI predicts is winning.
                
                #Draws the predicted points.
                for row, col in zip(*preds):
                    scale_val = int(9*moves[row, col])
                    pygame.draw.circle(screen,red,((margin + dim) * col + margin+2 ,(margin +dim)*row+margin+2),scale_val )
            
            
            #If there have been two passes then the game is ended.
            if self.passes ==2:
                
                done = True

            #Once a game is finished the winner of the game is displayed and then the game is closed after 10s.
            if done:
                
                screen.fill(self.black)
                
                points_winner = self.calculate_score() 
                
                self.draw_text(f'Winner is {self.colour_num[points_winner[0]]} by {points_winner[1]} points',self.white, screen, 170,262)
                
                pygame.display.flip()
                
                time.sleep(10)
                self.quit()
                
            clock.tick(60) #Setting the clock cycle to 60fps.
            pygame.display.flip()
            
