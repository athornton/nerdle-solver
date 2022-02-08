#!/usr/bin/env python3
"""
This is the CLI for the Nerdle solver.
"""

import argparse

from .nerdle_solver import NerdleSolver


def parse_args() -> argparse.Namespace:
    """
    Build and populate a CLI argument parser
    """
    parser = argparse.ArgumentParser(description="Solve a Nerdle puzzle")
    parser.add_argument(
        "-a", "--answer", help="the correct equation", default=""
    )
    parser.add_argument(
        "-d", "--debug", help="debug", action="store_true", default=False
    )
    parser.add_argument(
        "-g", "--guesses", help="number of guesses", default=6, type=int
    )
    parser.add_argument(
        "-i", "--initial-guess", help="initial guess", default=""
    )
    parser.add_argument(
        "-l", "--length", help="length of the expression", default=8, type=int
    )
    parser.add_argument(
        "-t",
        "--top",
        help="# of guesses to display (interactive)",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    return args


def generate_data() -> None:
    """
    Generate the equation caches
    """
    args = parse_args()
    for n in [6, 8]:  # pylint: disable="invalid-name"
        _ = NerdleSolver(expression_length=n, debug=args.debug)
        # That's it!  If the files aren't there, that will generate them.


def main() -> None:
    """
    Create an argument parser, and then start a solver with the right
    arguments.
    """
    args = parse_args()
    solver = NerdleSolver(
        answer=args.answer,
        debug=args.debug,
        expression_length=args.length,
        guesses=args.guesses,
        initial_guess=args.initial_guess,
        top=args.top,
    )
    solver.play()


if __name__ == "__main__":
    main()
