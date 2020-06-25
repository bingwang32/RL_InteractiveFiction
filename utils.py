import re


def pretty_print_state(state):
    '''Pretty print the text of a state in the interactive fiction game'''
    state = str(state)
    state = state.replace('\\\'', '\'')
    pattern = re.compile(r'\\n|b\'|b"')
    state = re.sub(pattern, ' ', state)
    state = state.strip('\'').strip('\"').strip()
    return state

def demo_n_games(agent, n_games=1000, mode='agent'):
    '''Run n game simulations and compute the average performance'''
    scores = []
    for _ in range(n_games):
        score = agent.demo_game(mode=mode, verbose=False)
        scores.append(score)
    return sum(scores)/n_games
