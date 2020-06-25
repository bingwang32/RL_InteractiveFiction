from jericho import *
import numpy as np
import random


class Agent():
    '''A Q-learning agent with TD(0) updates, compatible with Jericho's
    FrotzEnv for interactive fiction.
    '''

    def __init__(self, env, epsilon=0.1, alpha=1,
                 gamma=0.999, default_actions=['n', 's', 'e', 'w']):
        self.game = env  # Init as FrotzEnv("story_file.z3/5/8")
        self.epsilon = epsilon  # Rate of exploration
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor on future rewards
        # For edge case where self.game.get_valid_actions() = [];
        # e.g. A direction gets agent out of loop in Detective
        self.default_actions = default_actions
        self.V = dict()  # Q-table; self.V[state][action] = value
        self.valid_actions = dict()  # Cache of valid actions for states seen

    # ---------------------------------------------------------------------- #
    # Utility/helper/base methods:
    def get_state_pretty(self):
        '''Get state, return it as prettified, human-readable string
        using pretty_print_state() from utils
        '''
        return pretty_print_state(self.game.get_state()[-1])

    def get_valid_actions_memo(self):
        '''Memoize FrotzEnv's get_valid_actions() method to avoid repeated
        calls for previously seen states
        '''
        state = self.get_state_pretty()
        if state not in self.valid_actions.keys():
            available_actions = self.game.get_valid_actions()
            # For edge case: No valid actions found by Jericho's NLP functions
            if not available_actions:
                available_actions = self.default_actions
            self.valid_actions[state] = available_actions
        return self.valid_actions[state]

    def get_sa_value(self, state, action):
        '''Look up state-action value.
        If never seen state-action combo, then assume neutral (value = 0).
        '''
        if state in self.V.keys():
            if action in self.V[state].keys():
                return self.V[state][action]
        return 0

    def put_sa_value(self, state, action, value):
        '''Enter a value corresponding to a state and action into Q-table'''
        if state not in self.V.keys():
            self.V[state] = dict()
        self.V[state][action] = value

    def get_max_state_value(self, state):
        if state not in self.V.keys():
            return 0  # If state not seen yet, its max value is init val of 0
        else:
            return max(self.V[state].values())

    def get_best_action(self):
        '''Find best action when exploiting/maximizing expected value
        (greedy choice). Getting best action from self.V, the dictionary of
        values saved from learning
        '''
        state = self.get_state_pretty()
        max_val = self.get_max_state_value(state)
        available_actions = self.get_valid_actions_memo()
        max_val_actions = [a for a in available_actions
                           if self.get_sa_value(state, a) == max_val]
        return random.choice(max_val_actions)

    # ---------------------------------------------------------------------- #
    # Methods for reinforcement learning:
    def learn_select_action(self):
        '''Select best action with probability 1-epsilon (exploit),
        select random action with probability epsilon (explore)
        '''
        best_action = self.get_best_action()
        available_actions = self.get_valid_actions_memo()
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            return best_action

    def learn_from_action(self):
        '''Q-learn from a state and action; updates Q-table at every
        step (temporal difference TD(0) updates)
        '''
        state = self.get_state_pretty()
        action = self.learn_select_action()

        # Q(s, a): Current state-action value
        Q_sa = self.get_sa_value(state, action)
        state_prime, reward, _, _ = self.game.step(action)
        # max a  Q(s',a): Action that maxim. future value
        max_a_Q_sa = self.get_max_state_value(state_prime)

        # Update the Q-value:
        # New Q-val = current Q-val + (learn. rate)*(new Q-val - current Q-val)
        # Q(s, a) <- Q(s, a) + alpha*(r + gamma*(max a  Q(s',a)) - Q(s, a))
        new_Q_sa = Q_sa + self.alpha * (reward + self.gamma*max_a_Q_sa - Q_sa)
        self.put_sa_value(state, action, new_Q_sa)

    def learn_from_episode(self):
        '''For a full episode (from start to game over), learn with Q-learning
        at each move, updating the Q-table
        '''
        self.game.reset()
        while not self.game.game_over() and not self.game.victory():
            self.learn_from_action()

    def learn_game(self, n_episodes=1000, print_interval=100):
        '''Train the agent to play the game over n episodes'''
        for episode in range(1, n_episodes+1):
            self.learn_from_episode()
            if episode % print_interval == 0:
                print(f'Game #{episode} final score: {self.game.get_score()}')

    # ---------------------------------------------------------------------- #
    # Method for demoing game:
    def demo_game(self, mode='agent', verbose=True):
        self.game.reset()
        if mode == 'human':
            print('Welcome! Enter "I quit" at any point to exit the game.')

        # Full list of optimal steps to get max score in game
        walkthrough_actions = self.game.get_walkthrough()
        i = 0  # Iteration
        while not self.game.game_over() and not self.game.victory():
            # Play through game w/random actions at each step
            if mode == 'random':
                action = random.choice(self.get_valid_actions_memo())
            # Play through game w/optimal actions at each step (for max score)
            elif mode == 'walkthrough':
                action = walkthrough_actions[i]
            # Play through game w/best actions learned by agent
            # (Greedy choice with respect to saved value in Q-table)
            elif mode == 'agent':
                action = self.get_best_action()
            # Play through game with human inputs
            elif mode == 'human':
                print()
                print(self.get_state_pretty())
                available_actions = self.get_valid_actions_memo()
                input_msg = 'Choose one of the following valid moves, ' \
                            + 'or type in something else: ' \
                            + f'{available_actions} '
                action = input(input_msg)
                if action == 'I quit':
                    print()
                    exit_msg = 'Thanks for playing! Your final score was: ' \
                               + f'{self.game.get_score()}'
                    print(exit_msg)
                    return
            # If input anything else for mode, throw error
            else:
                msg = 'ERROR: The playthrough mode entered does not exist. ' \
                      + 'Enter "random" for a playthrough with random moves, '\
                      + '"walkthrough" for a playthrough with the optimal ' \
                      + 'steps to get the highest possible score, ' \
                      + '"agent" for a playthrough with the best moves ' \
                      + 'learned by the agent,' \
                      + 'or "human" to play through interactively.'
                return msg

            if verbose and mode != 'human':
                print(f'Iteration {i}')
                print(f'State: {self.get_state_pretty()}')
                print(f'Action selected: {action}')
            self.game.step(action)
            if verbose and mode != 'human':
                print(f'Total score after step: {self.game.get_score()}')
                print()
            i += 1

        if mode == 'human':
            print(self.get_state_pretty())
            print()
            if self.game.victory():
                end_msg = 'Game over! You won, and your final score was: ' \
                          + f'{self.game.get_score()}'
            else:
                end_msg = 'Game over! You lost, and your final score was: ' \
                          + f'{self.game.get_score()}'
            print(end_msg)
            return
        return self.game.get_score()
