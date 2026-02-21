# Rock Paper Scissors

This is my solution to the free code camp machine learning Rock Paper Scissors project ( https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors). The goal was to beat 4 bots with an accuracy of over 60%. I managed to have an average of 99% win rate against two of the bots, 85% and 68% for the other 2.

Strategy:
Instead of relying on a single algorithm, i built a multi-armed bandit ensemble. It uses 9 different agents, which predict the move that the opponet will play next.
Every round, the bot evaluates which of the 9 agents would have the best prediction rate in recent moves, using a decaying average (EMA)

The agents:
1. An online classifier machine learning model (Sgdclassifier with partial fit)
   -my last 5 moves
   - the opponent's last 5 moves
2. A randomized guesser, this exists to ensure a baseline accuracy when the other agents are fooled by the opponent and also
   to make time for the SGD classifier and the markov chain to learn.
3. The last move of the opponent
4. The counter to the their last move
5. The move that is countered by their last move
6. A 3rd order markov chain which uses my last move and the last 2 moves of the opponent and also has a decaying factor
7, 8 and 9 are similar to 3,4,5 except they use my last move, not the opponent's

Because opponets can randomly change their strategies in a game, this bot has a decaying average, which means it can quickly detect switches and choose the appropriate agent for their strategy. The SGD classifier and the markov chain can counter more complex strategies, while the other 6 (except the random answer) are three simple strategies an opponent can use.
Dependencies:
-numpy
-sklearn
