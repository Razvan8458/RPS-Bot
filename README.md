# Rock Paper Scissors

This is my solution to the free code camp machine learning Rock Paper Scissors project ( https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors)

I used an ensembly of 6 agents, which predict that move the opponent will play next:
1. An online classifier machine learning model (Sgdclassifier with partial fit)
   -my last 5 moves
   - the opponent's last 5 moves
2. A randomized guesser
3. The last move of the opponent
4. The counter to the their last move
5. The move that is countered by their last move
6. A markov chain which uses my last move and the last 2 moves of the opponent

Dependencies:
-numpy
-sklearn
