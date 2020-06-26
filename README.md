# Navigating Interactive Fiction with Reinforcement Learning
I trained a reinforcement learning agent to navigate an interactive fiction game through Q-learning.

Please see my detailed report [here](https://github.com/bingwang32/RL_InteractiveFiction/blob/master/Navigating_Interactive_Fiction_with_Reinforcement_Learning.ipynb) and my video presentation [here](https://www.youtube.com/watch?v=LJkoLr46280).

# Research Question
Can an agent learn to navigate an interactive fiction game with reinforcement learning?

# Overview and Background
<img src="images/michelle-ding-QAOtKq8ehcw-unsplash.jpg" width="850">

**What is interactive fiction?**<br>
It is a type of interactive story game, where users navigate the story by inputting text in response to text prompts.

**What are the game mechanics?**<br>
- Player reads a text description, and responds by inputting some text that makes sense 
    - In Dectective, these usually directions (N, S, E, W) or verb + noun combinations (e.g. “take paper”)
    - The game will respond to your input with text about what happens next/as a result of your action
- Player can get points throughout the game: Maximize cumulative points before game over
    - Not simple win/lose: Can still get a lot of points before player "loses" game

**Why is this a reinforcement learning problem?**<br>
Navigating interactive fiction is a problem that can be formulated as a reinforcement learning problem because it includes the following necessary components:
- Agent: Player
- Environment: The interactive story game and gameplay
- State: The text appearing at each stage in the story
- Action: The text a player inputs in response to each stage's descriptive text
- Reward: Points acquired/lost in response to a player's actions at a given stage

**What were the tools used in this analysis?**<br>
I used [Jericho](https://github.com/microsoft/jericho), an environment connecting interactive fiction games to reinforcement learning agents. It was written by some folks at Microsoft, Matthew Hausknecht and Prithviraj Ammanabrolu, for their 2019 paper, [Interactive Fiction Games: A Colossal Adventure](https://arxiv.org/abs/1909.05398). Its environment, FrotzEnv, had useful methods for a reinforcement learning agent:
- For instance, `get_valid_actions()` uses natural language processing (NLP) to generate valid actions from a given state.

**How was reinforcement learning used in this project?**<br>
I used *Q-learning* to train my agent. Q-learning worked for this problem because it is:
- *Model-free:* Great because we don’t know states and actions ahead of time. At the start of the game, the agent only sees the first state its corresponding valid actions
- *Temporal Difference TD(0) Updates:* Helpful because states can repeat and it’s best to update after every step instead of at the end of the episode. You can go in cycles: For instance, you can go from a main hallway into a room and come back out to the hallway.
- *Maximizes expected cumulative reward:* Same goal as the goal of our game. 
- *Off-policy learning:* Learn from policies not used in gameplay

# Model and Results

