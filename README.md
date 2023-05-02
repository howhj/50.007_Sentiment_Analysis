# 50007-Sentiment-Analysis

This is the group project for the term 6 50.007 course. The goal was to try various algorithms to perform sentiment analysis, with the main focus on hidden Markov models.

The project is split into 4 parts:
1. A basic sentiment analysis model using emission probabilities only
2. 1st order hidden Markov model
3. 2nd order hidden Markov model
4. Anything goes

For part 4, we were allowed to implement any algorithm we wanted, with restrictions on which external libraries we could use. Due to a lack of time, we ended up extending our hidden Markov model to the 3rd order for part 4.

One possible improvement is to clean up the input data (e.g. lemmatisation) before feeding it into the model.

## Requirements

The following packages were used for this project:

- Python 3.10.10
- Numpy 1.24.2

## Running the Code

To run the code, enter the following command in a CLI:

    python p[x].py [k] [training file] [test input file] [output file for results]

where `[x]` is the number e.g. `p1.py`, `p2.py`.


You can also run the following command for more details:

    python p[x].py -h
