# CS 170 Project Spring 2020

cd into cs170-finalproject/ directory

Command to run program: for i in {1..10000}; do for file in inputs/*;do for j in {5,10,20,30,50,100}; do python3 solver.py $file $j;done;done;done;

j refers to the range of random values to add to the edge weights for solve_random_MST_cut_sometimes and solve_random_MST_cut_all

Note that many of our outputs come from this randomized algorithm, so they may take a long time to replicate. That's why these commands are run 10,000 times. 