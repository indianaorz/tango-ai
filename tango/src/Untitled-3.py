

we need to build a model which takes in the important information from the game and outputs the input commands based on it.

here is a high level architecture:





input: gamestate x memory
output [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 16 probabilities for pressing 16 possible buttons. multiple buttons can be pressed at a time

the model should be able to take in x amount of game states which we can increase or decrease for learning



here is the gamestate:

    cust gage = single float from 0-1
    grid 6x3 array of grid objects

    grid object has 4 properties
        grid_type = one hot encoded for values from 0-12
        grid_owner = single float from 0-1 (who owns the grid tile, player or enemy)
        player = single float from 0-1 (player is on this grid tile)
        enemy = single float from 0-1 (enemy is on this grid tile)
    
    player health = single float from 0-1
    enemy health = single float from 0-1
    
    player_chip = one hot encoded for values from 0-400
    enemy_chip = one hot encoded for values from 0-400
    
    player_chip_hand = list of 5 one hot encoded values from 0-400
    
    player_folder = list of 30 folder_chips
    enemy_folder = list of 30 folder_chips
    
    player_emotion_state = one hot encoded for values from 0-26
    enemy_emotion_state = one hot encoded for values from 0-26
    
    player_used_crosses = array of 10 floats from 0-1
    enemy_used_crosses = array of 10 floats from 0-1

    folder_chip has 4 properties
        id = one hot encoded for values from 0-400
        code = one hot encoded for values from 0-26
        used = single float from 0-1
        regged = single float from 0-1
        tagged = single float from 0-1
        
    player_custom = a list of 200 floats which can be 0 or 1
    enemy_custom = a list of 200 floats which can be 0 or 1
    
the model should have a scale parameter it can use to multiply the hidden layers for scaling and a memory parameter which can be used to increase the gamestates

we should make sure the model has approprite dropout for training and batch normalization and supports training in batches