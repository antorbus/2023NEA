def get_path_from_file(path_str): #Helper function that takes a path a string and returns the parent directory path.
    
    slash_pos = [pos for pos, char in enumerate(path_str) if char == '/'][-1] #Finds position of last / .
    
    return path_str[:slash_pos+1] #Returns truncated path.
