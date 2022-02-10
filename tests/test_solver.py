"""
Unit tests for the Wordle solver
"""

import pytest

from nerdle_solver.exceptions import OutOfGuesses
from nerdle_solver.nerdle_solver import NerdleSolver

# pylint: disable="invalid-name"


def test_minimal_class() -> None:
    """
    Test basic class defaults
    """
    n = NerdleSolver()
    assert n.debug is False
    assert n.expression_length == 8
    assert n.top == 5
    assert n.max_guesses == 6
    #
    # Check that the lexicon is right
    #
    assert len(n.remaining_possibilities) == 17774


def test_variant_class() -> None:
    """
    Test that non-default values get correctly set
    """
    n = NerdleSolver(debug=True, expression_length=6, top=3, guesses=5)
    assert n.debug is True
    assert n.expression_length == 6
    assert n.top == 3
    assert n.max_guesses == 5
    #
    # And the length of our valid list should change
    #
    assert len(n.remaining_possibilities) == 206


def test_solution() -> None:
    """
    Test that we solve a problem
    """
    n = NerdleSolver(initial_guess="8+7-2=13", answer="4*44=176")
    n.play()
    assert n.current_pattern == "!!!!!!!!"
    assert n.guesses == 4


def test_easy_solution() -> None:
    """
    Test that we solve an easy problem
    """
    n = NerdleSolver(
        expression_length=6, initial_guess="18/9=2", answer="5+8=13"
    )
    n.play()
    assert n.current_pattern == "!!!!!!"
    assert n.guesses == 3


@pytest.mark.xfail(raises=OutOfGuesses)
def test_failure() -> None:
    """
    Test that we run out of guesses appropriately
    """
    n = NerdleSolver(initial_guess="8+7-2=13", answer="4*44=176", guesses=3)
    n.play()


def test_internal_state() -> None:
    """
    Test that the elimination process works and that internal variables
    are correctly set
    """
    n = NerdleSolver(initial_guess="8+7-2=13", answer="4*44=176")
    assert len(n.remaining_possibilities) == 17774
    n.loop_once()
    assert len(n.remaining_possibilities) == 161
    n.loop_once()
    assert len(n.remaining_possibilities) == 2
    n.loop_once()
    assert len(n.remaining_possibilities) == 1
    assert n.in_expr == set("4*=176")
    assert n.not_in_expr == set("023589+-/")
