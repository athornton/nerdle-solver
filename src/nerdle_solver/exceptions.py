"""
Custom exceptions for the Nerdle solver
"""


class OutOfGuesses(Exception):
    """
    We ran out of guesses before finding the answer
    """


class OutOfEquations(Exception):
    """
    This shouldn't happen: if we ran out of equations there's a bug in the
    matcher
    """


class CorrectAnswer(Exception):
    """
    Nailed it.
    """
