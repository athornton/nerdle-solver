#!/usr/bin/env python3
"""
Walk through the dictionary: we're going to test all Nerdle
answers, with and record how many guesses each one took, whether we
solved the problem, for what fraction of words we managed to solve the
problem, median turns to solve, modal turns to solve, and mean turns
to solve the ones we managed to solve.
"""

import argparse
import logging
from dataclasses import dataclass
from statistics import mean, median, mode, pstdev
from typing import Optional, Union

from nerdle_solver.exceptions import OutOfGuesses
from nerdle_solver.nerdle_solver import NerdleSolver

# pylint: disable="logging-fstring-interpolation"


@dataclass
class NerdleResult:
    """
    Class for solution for a given word
    """

    word: str
    solved: bool
    guesses: Optional[int]


class NerdleStats:
    """
    Class for statistics across a result list.  Mean, median, and pstdev
    are calculated across successful results.  Mode is calculated across all
    results.
    """

    def __init__(
        self,
        name: str = "",
        results: list[NerdleResult] = [],
        initial_guess: str = "",
        debug=False,
    ):  # pylint: disable="dangerous-default-value"
        self.debug = debug
        self.log = logging.getLogger(__name__)
        level = logging.INFO
        if self.debug:
            level = logging.DEBUG
        logging.basicConfig(level=level)
        self.log.setLevel(level)
        self.name = name
        self.initial_guess = initial_guess
        self.results = results

    @property
    def successes(self) -> list[str]:
        """
        Return those results where the solver succeeded
        """
        return [x for x in self.results if x.solved]

    @property
    def failures(self) -> list[str]:
        """
        Return those results where the solver failed
        """
        return [x for x in self.results if not x.solved]

    @property
    def success_rate(self) -> float:
        """
        Return the ratio of successes to total attempts
        """
        return len(self.successes) / len(self.results)

    @property
    def failure_rate(self) -> float:
        """
        Return the ratio of failures to total attempts
        """
        return len(self.failures) / len(self.results)

    @property
    def median(self) -> float:
        """
        Return the median of successful attempts
        """
        return median([x.guesses for x in self.successes])

    @property
    def mean(self) -> float:
        """
        Return the mean of successful attempts
        """
        return mean([x.guesses for x in self.successes])

    @property
    def mode(self) -> float:
        """
        Return the mode of all attempts
        """
        return mode([x.guesses for x in self.results])

    @property
    def pstdev(self) -> float:
        """
        Return the population standard deviation of successful attempts
        """
        return pstdev([x.guesses for x in self.successes])

    @property
    def mad(self) -> float:
        """
        Return the mean absolute deviation of successful attempts
        """
        return mean([abs(self.mean - x.guesses) for x in self.successes])

    def report(self) -> None:
        """
        Report statistics across all attempts
        """
        self.log.info(f"{self.name} statistics:")
        self.log.info(
            f"Word count: {len(self.results)} ; "
            + f"Initial word: '{self.initial_guess}'"
        )
        self.log.info(
            f"Solved: {len(self.successes)} ; "
            + f"Failed: {len(self.failures)}"
        )
        self.log.info(f"Success rate: {100 * self.success_rate:02f}%")
        self.log.info(
            f"Mean guesses: {self.mean:03f} ; "
            + f"Median guesses: {int(self.median)}"
        )
        self.log.info(f"Mode of guesses: {self.mode}")
        self.log.info(f"Population std. dev (guesses): {self.pstdev:03f}")
        self.log.info(f"Mean population absolute deviation: {self.mad:03f}")


class NerdleTester:
    """
    Harness to test the solver performance across all legal equations
    """

    # pylint: disable="too-many-instance-attributes"
    def __init__(
        self,
        length: int = 8,
        initial_guess: str = "",
        game: str = "Nerdle",
        debug: bool = False,
    ):
        self.debug = debug
        self.log = logging.getLogger(__name__)
        level = logging.INFO
        if self.debug:
            level = logging.DEBUG
        logging.basicConfig(level=level)
        self.log.setLevel(level)
        self.length = length
        self.game = game
        self.log.debug(f"Creating {self.game} Tester class")
        self.initial_guess = initial_guess
        self.results: list[NerdleResult] = []
        self.stats: dict[str, Union[str, float]]
        ndl = NerdleSolver(expression_length=self.length)
        self.answers: list[str] = list(ndl._valid_equations.keys())
        self.statistics: NerdleStats = NerdleStats(
            name=self.game, initial_guess=self.initial_guess
        )
        self.count: int = 0

    def run(self) -> None:
        """
        Test all legal equations
        """
        for word in self.answers:
            self.try_word(word)
        self.process_statistics()

    def try_word(self, word: str) -> None:
        """
        Get results for a particular equation
        """
        self.count += 1
        ndl = NerdleSolver(
            debug=self.debug,
            answer=word,
            expression_length=self.length,
            initial_guess=self.initial_guess,
        )
        if not self.debug:
            # We don't want to see its guesses as they go.
            ndl.log.setLevel(logging.WARNING)
        result = NerdleResult(word=word, solved=False, guesses=None)
        r_strl = f"{self.game} {self.count}/{len(self.answers)}"
        r_strr = f"/{ndl.max_guesses} ({word})"
        try:
            ndl.play()
            result.solved = True
            result.guesses = ndl.guesses
            resultstr = f"{r_strl} {ndl.guesses}{r_strr}"
        except OutOfGuesses:
            resultstr = f"{r_strl} X{r_strr}"
        self.results.append(result)
        self.log.info(resultstr)

    def process_statistics(self) -> None:
        """
        There's not a lot to do, because it all lives in the class definitions
        """
        self.statistics.results = self.results
        self.statistics.report()


def parse_args() -> argparse.Namespace:
    """
    Build and populate a CLI argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate stats for Nerdle solver"
    )
    parser.add_argument(
        "-d", "--debug", help="debug", action="store_true", default=False
    )
    parser.add_argument(
        "-g", "--game", "--game-name", help="name of game", default="Nerdle"
    )
    parser.add_argument(
        "-i",
        "--initial",
        "--initial-guess",
        help="initial guess",
        default="7+8-3=12",
    )
    parser.add_argument(
        "-l",
        "--length",
        "--expression-length",
        help="length of equation in characters",
        default=8,
        type=int,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Create argument parser, parse the args, and loop over games"""
    args = parse_args()
    tester = NerdleTester(
        debug=args.debug,
        game=args.game,
        initial_guess=args.initial,
        length=args.length,
    )
    tester.run()


if __name__ == "__main__":
    main()
