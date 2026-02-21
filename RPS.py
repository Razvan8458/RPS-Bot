import numpy as np

from sklearn.linear_model import SGDClassifier

import random

def transform(prev_play):

    if prev_play == 'R':

        return 0

    elif prev_play == 'S':

        return 1

    else:
        return 2

counter = {0: 'P', 1: 'R', 2: 'S'}

def agent_1(state = {}, my_history=[], opponent_history=[]):
    # The first agent is an online classification machine learning model
    # it uses my last 5 moves and the opponets 5 last move to predict the next move
    # of the opponent
    # I chose an SGDClassifier, so that i can use its partial fit
    output = agent_2()
    if len(opponent_history) > 6:
        
        X = np.array([[my_history[-6], my_history[-5], my_history[-4], my_history[-3], my_history[-2], opponent_history[-6], opponent_history[-5], opponent_history[-4], opponent_history[-3], opponent_history[-2]]])

        y = np.array([opponent_history[-1]])
        if state['first_time'] == 1:

            state['model'].partial_fit(X, y, classes = [0, 1, 2])
            state['first_time'] = 0

        else:

            state['model'].partial_fit(X, y)

        X = np.array([[my_history[-5], my_history[-4], my_history[-3], my_history[-2], my_history[-1], opponent_history[-5], opponent_history[-4], opponent_history[-3], opponent_history[-2], opponent_history[-1]]])

        output = state['model'].predict(X)[0]
    return output
def agent_2():
    # The second agent randomly chooses what it thinks the opponent will paly
    random_list = [0, 1, 2]
    output = random.choice(random_list)
    return output
def agent_3(history=[]):
    # The third agent predicts the opponent will play the same move as the last
    # played by the opponent / me
    if len(history) < 1:
        return agent_2()
    return history[-1]
def agent_4(history=[]):
    # The fourth agent predicts the opponent will play the move that counters
    # the opponents / my last move
    if len(history) < 1:
        return agent_2()
    return transform(counter[history[-1]])
def agent_5(history=[]):
    # The fifth agent predicts the opponent will play the move that is counter
    # by the opponents / my last move
    # That is equivelant to the the move that counters the counter of the move
    if len(history) < 1:
        return agent_2()
    return transform(counter[transform(counter[history[-1]])])
def agent_6(state = {}, opponent_history=[], my_history=[]):
    # The sixth agent predicts the move of the opponent based on a markov chain
    # the markov chain uses my last move and the last two moves of the opponent
    # i also used a value to multiply the former moves, because i want it to focus
    # a little more on more current moves
    # i didnt make a complete markov chain, because it doesnt take probability
    # but the number of times a change happened, with a decay for moves that happened
    # previously
    output = agent_2()
    if len(opponent_history) > 3:
        former_state = my_history[-2] * 9 + opponent_history[-2] * 3 + opponent_history[-3]
        former_prob = state['markov_chain'][former_state][opponent_history[-1]] 
        value = 0.95
        state['markov_chain'][former_state][opponent_history[-1]] = former_prob * value + 1
        current_state = my_history[-1] * 9 + opponent_history[-1] * 3 + opponent_history[-2]
        best_one = 0
        if state['markov_chain'][current_state][1] > state['markov_chain'][current_state][best_one]:
            best_one = 1
        if state['markov_chain'][current_state][2] > state['markov_chain'][current_state][best_one]:
            best_one = 2
        output = best_one
    return output
def player(prev_play, state = {}, my_history=[], opponent_history=[]):

    if prev_play == '':
        opponent_history.clear()
        my_history.clear()
        state.clear()

    opponent_history.append(transform(prev_play))

    if not state:
        state['first_time'] = 1
        state['model'] = SGDClassifier(random_state = 42)
        state['nr_correct'] = [0.5] * 9
        state['outputs'] = [-2] * 9
        state['markov_chain'] = [[0 for i in range(3)] for j in range(27)]
    output_1 = agent_1(state, my_history, opponent_history)
    output_2 = agent_2()
    output_3 = agent_3(opponent_history)
    output_4 = agent_4(opponent_history)
    output_5 = agent_5(opponent_history)
    output_6 = agent_6(state, opponent_history, my_history)
    output_7 = agent_3(my_history)
    output_8 = agent_4(my_history)
    output_9 = agent_5(my_history)

    best_one = 0
    for i in range(0,9):
        ok = 0
        if state['outputs'][i] == transform(prev_play):
            # this means the agent would have guessed correctly last time
            ok = 1
        value = 0.1
        state['nr_correct'][i] = ok * value + (1 - value) * state['nr_correct'][i]
        if state['nr_correct'][i] > state['nr_correct'][best_one]:
            best_one = i
        # i make a corectness stat for each agent
        # once again, i make it have a decay for former correct moves
    state['outputs'] = [output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9]
    if len(opponent_history) < 10:
        # for the first ten moves i let only the random agent choose
        output = output_2
    else:
        # after ten moves i pick the agent that performed the best recently
        output = state['outputs'][best_one]
    # after i have predicted what move the opponet will play
    # i choose the move that counters it
    guess = counter[output]

    my_history.append(transform(guess))

    return guess