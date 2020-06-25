from jericho import *
import numpy as np
import random


class Agent():
    '''A Q-learning agent with TD(0) updates, compatible with Jericho's
    FrotzEnv for interactive fiction.
    '''

    def __init__(self, env, epsilon=0.1, alpha=1, gamma=0.999, default_actions=['n','s','e','w']):
        self.game = env  # Need to initialize as FrotzEnv("path-to-game-file.z3/5/8"); .reset() for new game
        self.epsilon = epsilon
        self.alpha = alpha  # Learning rate; proportion of updated Q-value that consists of the new Q-value
        self.gamma = gamma  # Discount factor on future rewards
        self.default_actions = default_actions  # For edge case where self.game.get_valid_actions() = [];
                                                # e.g. A direction gets agent out of loop in Detective
        self.V = dict() # Build up the values of different states as we encounter them; Note the Markov assumption
        self.valid_actions = dict()  # Cache of valid actions to previously seen states

    # ---------------------------------------------------------------------------------------------------- #
    ## Utility/helper/base methods:

    def get_state_pretty(self):
        '''Get state, return it as prettified, human-readable string using pretty_print_state'''
        return pretty_print_state(self.game.get_state()[-1])

    def get_valid_actions_memo(self):
        '''Memoize FrotzEnv's get_valid_actions() method to avoid repeated calls for previously seen states'''
        state = self.get_state_pretty()
        if state not in self.valid_actions.keys():
            available_actions = self.game.get_valid_actions()
            if not available_actions: available_actions = self.default_actions  # For edge case of no valid actions
            self.valid_actions[state] = available_actions
        return self.valid_actions[state]

    def get_sa_value(self, state, action):
        '''Look up state-action value. If never seen state-action combo, then assume neutral.'''
        if state in self.V.keys():
            if action in self.V[state].keys():
                return self.V[state][action]
        return 0

    def put_sa_value(self, state, action, value):
        if state not in self.V.keys():
            self.V[state] = dict()
        self.V[state][action] = value

    def get_max_state_value(self, state):
        if state not in self.V.keys():  # If state not encountered yet, its max value is initial val of 0
            return 0
        else:
            return max(self.V[state].values())

    def get_best_action(self):
        '''Find best action when exploiting/maximizing expected value (greedy choice).
        Getting best action from self.V, the dictionary of values saved from learning
        '''
        state = self.get_state_pretty()
        max_val = self.get_max_state_value(state)
        available_actions = self.get_valid_actions_memo()
        max_val_actions = [a for a in available_actions if self.get_sa_value(state, a)==max_val]
        return random.choice(max_val_actions)

    # ---------------------------------------------------------------------------------------------------- #
    ## Methods for reinforcement learning:

    def learn_select_action(self):
        '''Select best action with probability 1-epsilon (exploit), select random action
        with probability epsilon (explore)
        '''
        best_action = self.get_best_action()
        available_actions = self.get_valid_actions_memo()
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            return best_action

    def learn_from_action(self):
        "Q-learning"
        state = self.get_state_pretty()  # s: Current state
        action = self.learn_select_action()

        Q_sa = self.get_sa_value(state, action)  # Q(s, a): Current state value. self.V[state][action]
        # NOTE: This steps the game forward to the next state!! Don't step again!
        state_prime, reward, _, _ = self.game.step(action)
        max_a_Q_sa = self.get_max_state_value(state_prime)  # max a  Q(s',a): Action that maxim. future value

        # Update the Q-value with the current Q-value + (learning rate)*(new Q-value - current Q-value)
        # Q(s, a) <- Q(s, a) + alpha*(r + gamma*(max a  Q(s',a)) - Q(s, a))
        new_Q_sa = Q_sa + self.alpha*(reward + self.gamma*max_a_Q_sa - Q_sa)
        self.put_sa_value(state, action, new_Q_sa)

    def learn_from_episode(self):
        "Update Values based on reward."
        self.game.reset()
        while not self.game.game_over() and not self.game.victory():
            self.learn_from_action()
        # NOTE: The line below is to update the value in V for last state, after the game ends.
        # Since we get points throughout the game, and since (e.g. in Detective) you get a lot of points for
        # winning the game, and value updates for those points are captured prior to the terminal state,
        # we do not need to save a reward for the terminal state.
        # If we do, the reward is likely 0
        # (or current state's .get_score() - previous state's .get_score(); configure in learn_from action)
        #self.V[self.get_state_pretty()][''] = 0

    def learn_game(self, n_episodes=1000, print_interval=100):
        "Let's learn through complete experience to get that reward."
        for episode in range(1, n_episodes+1):
            self.learn_from_episode()
            if episode % print_interval == 0:
                print(f'Game #{episode} final score: {self.game.get_score()}')

    # ---------------------------------------------------------------------------------------------------- #
    ## Methods for demoing game:

    def demo_game(self, mode='agent', verbose=True):
        self.game.reset()
        if mode == 'human':
            print('Welcome! Enter "I quit" at any point to exit the game.')

        walkthrough_actions = self.game.get_walkthrough()  # Full list of optimal steps to get max score in game
        i = 0  # Iteration
        while not self.game.game_over() and not self.game.victory():
            # Play through game with random actions at each step
            if mode == 'random':
                action = random.choice(self.get_valid_actions_memo())
            # Play through game with optimal actions at each step (to get max score)
            elif mode == 'walkthrough':
                action = walkthrough_actions[i]
            # Play through game with best actions as learned by agent via Q-learning
            elif mode == 'agent':
                action = self.get_best_action()
            # Play through game with human inputs
            elif mode == 'human':
                print()
                print(self.get_state_pretty())
                available_actions = self.get_valid_actions_memo()
                action = input(f'Choose one of the following valid moves, or type in something else: {available_actions} ')
                if action == 'I quit':
                    print()
                    print(f'Thanks for playing! Your final score was: {self.game.get_score()}')
                    return
            # If input anything else for mode, throw error
            else:
                error_msg = 'ERROR: The playthrough mode entered does not exist. ' \
                            + 'Enter "random" for a playthrough with random moves, ' \
                            + '"walkthrough" for a playthrough with the optimal steps to get the ' \
                            + 'maximum possible score for the game, ' \
                            + 'or "agent" for a playthrough with the best moves learned by the agent.'
                return error_msg

            if verbose == True and mode!='human':
                print(f'Iteration {i}')
                print(f'State: {self.get_state_pretty()}')
                print(f'Action selected: {action}')
            self.game.step(action)
            if verbose == True and mode!='human':
                print(f'Total score after step: {self.game.get_score()}')
                print()
            i += 1

        if mode == 'human':
            print(self.get_state_pretty())
            print()
            if self.game.victory():
                print(f'Game over! You won, and your final score was: {self.game.get_score()}')
            else:
                print(f'Game over! You lost, and your final score was: {self.game.get_score()}')
            return
        return self.game.get_score()
