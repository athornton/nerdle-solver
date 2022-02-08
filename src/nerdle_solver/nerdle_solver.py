#!/usr/bin/env python3
"""
Play Nerdle (https://nerdlegame.com)
"""
import json
import logging
from itertools import product
from os import mkdir
from os.path import dirname, join, realpath
from typing import Any, Iterator, List, Optional, Tuple

from .exceptions import CorrectAnswer, OutOfEquations, OutOfGuesses


class NerdleSolver:
    """
    Encapsulates the logic of Nerdle (https://nerdlegame.com)
    """

    def __init__(
        self,
        answer: str = "",
        debug: bool = False,
        expression_length: int = 8,
        guesses: int = 6,
        initial_guess: str = "",
        top: int = 5,
        expr_file: str = "",
    ):
        self.debug = debug
        self.log = logging.getLogger(__name__)
        level = logging.INFO
        if self.debug:
            level = logging.DEBUG
        logging.basicConfig(level=level)
        self.log.setLevel(level)
        self.valid_guess: str = ""
        self.guess: str = initial_guess
        self.answer: str = answer
        self.expr_value: int = -1
        self.expression_length: int = expression_length
        self.current_pattern: str = ""
        self.guesses: int = 1
        self.max_guesses: int = guesses
        self.top: int = top
        self.known_character_positions: str = "X" * self.expression_length
        self.position_could_be: List(set[str]) = []
        self.legal_chars: str = "0123456789+-*/="
        for n in range(self.expression_length):
            self.position_could_be.append(set([c for c in self.legal_chars]))
        self.not_in_expr: set[str] = set()
        self.in_expr: set[str] = set()
        self.guesses_tried: list[str] = []
        # in self._expr_by_str, a value of None means the expression does not
        # parse.  This lets us cache failed parses as well.
        self._expr_by_str: dict[str, Optional[int]] = dict()
        self._expr_by_val: dict[int, List[str]] = dict()
        self._valid_equations: dict[str, int] = dict()
        expr_loaded = False
        if not expr_file:
            datadir = realpath(join(dirname(__file__), "static"))
            expr_file = f"{datadir}/expressions-{self.expression_length}.json"
        try:
            with open(expr_file, "r") as f:
                _exprs = json.load(f)
                self._expr_by_str = _exprs["by_str"]
                self._expr_by_val = _exprs["by_val"]
                self._valid_equations = _exprs["valid"]
            expr_loaded = True
        except Exception as exc:
            self.log.debug(f"Failed to read expr file {expr_file}: {exc}")
            self.log.debug("Calculating legal expressions")
            self.generate_legal_expressions()
        if not expr_loaded:
            try:
                self.log.debug("Writing expression cache")
                try:
                    mkdir(datadir)
                except FileExistsError:
                    pass
                d = {
                    "by_str": self._expr_by_str,
                    "by_val": self._expr_by_val,
                    "valid": self._valid_equations,
                }
                with open(expr_file, "w") as f:
                    json.dump(d, f)
            except Exception as exc:
                self.log.debug(f"Failed to write expr file {expr_file}: {exc}")
        self.remaining_possibilities: list[str] = list(
            self._valid_equations.keys()
        )
        self.sort_remainder()
        if not self.guess:
            self.log.debug(
                "Best initial guesses:"
                + f"{self.remaining_possibilities[:self.top]}"
            )

    def play(self) -> None:
        """
        Main loop of the game
        """
        try:
            while True:
                if self.guesses > self.max_guesses:
                    raise OutOfGuesses()
                self.loop_once()
        except CorrectAnswer as exc:
            print(f"Correct answer: '{exc}' in {self.guesses} guesses")
            return

    def loop_once(self) -> None:
        """
        Single pass through the loop
        """
        self.choose_or_show_next_guess()
        self.get_current_guess()
        self.update_pattern()
        self.guess = ""
        self.valid_guess = ""
        self.restrict_possibilities()
        self.sort_remainder()

    def get_current_guess(self) -> None:
        """
        Get and check a guess
        """
        while not self.valid_guess:
            self.solicit_current_guess()
            self.check_current_guess()
        self.guesses += 1

    def solicit_current_guess(self) -> None:
        """
        Interactively get a guess
        """
        self.guess = input("Guess expression > ")

    def check_current_guess(self) -> None:
        """
        Is the supplied guess valid?
        """
        try:
            self.validate_guess(self.guess)
            self.valid_guess = self.guess
        except ValueError as exc:
            self.log.debug(f"'{self.guess}' failed: {exc}")
            self.guess = ""
            self.expr_value = -1

    def parse_expr(self, expr: str) -> int:
        """
        This is the central feature.  Take a string consisting of digits and
        operators (excluding '='), and try to reduce it to an integer.

        If it succeeds, it stores the result in the self._expr_by_str cache,
        and if it fails, it stores None in that cache.
        """
        if expr in self._expr_by_str:
            if self._expr_by_str[expr] is None:
                raise ValueError(f"'{expr}' is known not to parse")
            return self._expr_by_str[expr]
        ttok = []
        curstr = ""
        for c in expr:
            if c.isdigit():
                curstr += c
            else:
                try:
                    # This will catch both leading zeroes and repeated
                    # operators
                    ttok.append(self.check_is_valid_number(curstr))
                except ValueError:
                    # Mark as invalid
                    self.store_expr(expr, None)
                    raise
                curstr = ""
                ttok.append(c)
        if curstr:
            # The last token was a number
            ttok.append(self.check_is_valid_number(curstr))
        # Now ttok contains alternating ints and strings representing
        # operations
        while True:
            if len(ttok) == 1:
                if ttok[0] < 0:
                    raise ValueError("Only non-negative results allowed")
                break
            for idx, tok in enumerate(ttok):
                if isinstance(tok, int):
                    continue
                if tok == "*" or tok == "/":  # high-priority operator
                    if tok == "/":
                        gazinta = ttok[idx - 1]
                        gazunda = ttok[idx + 1]
                        try:
                            quotient = gazinta / gazunda
                        except ZeroDivisionError:
                            # Mark as invalid
                            self.store_expr(expr, None)
                            raise ValueError("Attempted division by zero")
                        if quotient != int(quotient):
                            # Mark as invalid
                            self.store_expr(expr, None)
                            raise ValueError(
                                f"{gazunda} doesn't integrally "
                                + f"divide {gazinta}"
                            )
                        result = int(quotient)  # Python 3 division is float!
                    else:
                        result = ttok[idx - 1] * ttok[idx + 1]
                else:
                    if tok == "+":
                        result = ttok[idx - 1] + ttok[idx + 1]
                    else:
                        result = ttok[idx - 1] - ttok[idx + 1]
                # Replace the numbers on either side of the operator,
                #  and the operator itself, with the result.  Restart
                #  parsing ttok.
                first = []
                last = []
                if idx > 2:
                    first = ttok[: idx - 1]
                if len(ttok) > idx + 1:
                    last = ttok[idx + 2 :]
                ttok = first + [result] + last
                break  # From the inner for, not the 'while True'
        lhs = ttok[0]
        return lhs

    def validate_guess(self, guess) -> None:
        """
        Only returns if guess is plausible; raises ValueError otherwise
        """
        chars_in_guess = set(guess)
        if chars_in_guess < self.in_expr:
            raise ValueError(
                f"{self.in_expr} are all in the expression, but "
                + f"{self.guess} only has {chars_in_guess}"
            )
        for idx, c in enumerate(guess):
            if c in self.not_in_expr:
                raise ValueError(f"'{c}' is known to not be in expression")
            if c not in self.position_could_be[idx]:
                raise ValueError(
                    f"'{c}' cannot be in position {idx}: "
                    + f"not one of {self.position_could_be[idx]}"
                )
        # Well, it *could* be right.

    def update_pattern(self) -> None:
        """
        If we know the answer, figure out the pattern; if not, request it
        from the user (who's presumably getting it from the game)
        """
        if self.answer:
            self.calculate_pattern()
        else:
            self.solicit_pattern()
        if self.current_pattern == "!" * self.expression_length:
            raise CorrectAnswer(self.guess)
        self.update_positions()

    def calculate_pattern(self) -> None:
        """
        If we know the answer, generate the response pattern
        """
        pattern = ""
        assert self.answer, "Cannot calculate pattern without the answer"
        for idx, c in enumerate(self.valid_guess):
            self.log.debug(f"considering '{c}' in position {idx}")
            p = self.answer[idx]
            if c == p:
                self.log.debug(f"'{c}' is in position {idx}")
                pattern += "!"
            elif c not in self.answer:
                self.log.debug(f"'{c}' does not appear in expression")
                pattern += "."
            else:
                self.log.debug(
                    f"'{c}' appears in expression, but not in position {idx}"
                )
                pattern += "D"
        # Just like update_positions, we do a second pass to catch multiples
        #  where we already have them all
        for idx, c in enumerate(self.valid_guess):
            if pattern[idx] != "D":
                continue
            actual_count = self.answer.count(c)
            # How many do we have that we know where they are?
            # There's gotta be a better way to do this, but let's get it
            # working first.
            pattern_count = 0
            for a_idx, a_c in enumerate(self.answer):
                if a_c == c:
                    if pattern[a_idx] == "!":
                        pattern_count += 1
            assert pattern_count <= actual_count, f"Overcount of '{c}'"
            # This might not be stable.
            pattern_char = "?"  # Default: we don't know where they all are
            if pattern_count == actual_count:
                self.log.debug(f"Already found all occurrences of '{c}'")
                pattern_char = "."
            else:
                self.log.debug(f"'{c}' appears but position unknown")
            # This should just replace this "D" with a resolved "?" or "."
            pattern = pattern[:idx] + pattern_char + pattern[idx + 1 :]
        self.current_pattern = pattern

    def solicit_pattern(self) -> None:
        """
        Since we don't know the answer, ask about the pattern
        """
        while True:
            response = input("Response pattern > ")
            if len(response) != self.expression_length:
                continue
            rchars = set(response)
            if not rchars <= set("!?."):
                self.log.debug(f"rchars {rchars}; {set('!?.')}")
                continue
            self.current_pattern = response
            break

    def update_positions(self) -> None:
        """
        For each position in the expression, update the set of possible
        characters
        """
        self.guesses_tried.append(self.valid_guess)
        for idx, c in enumerate(self.current_pattern):
            g = self.valid_guess[idx]
            setc = set(g)
            if c == "!":
                self.position_could_be[idx] = setc  # Fixed in place
                self.in_expr |= setc
                self.log.debug(f"position {idx}: '{g}'")
                continue
            if c == "?":
                self.position_could_be[idx] ^= setc
                self.in_expr |= setc
                self.log.debug(f"position {idx}: not '{g}'")
                self.log.debug(f"'{g}' in expression")
        # Now we start over.  This catches the case of "not in word" that
        #  really means "it's a multiple, and you have too many, and it's
        #  not here" because by the time we do this, if we have any
        #  occurrences, they will be in self.in_expr
        for idx, c in enumerate(self.current_pattern):
            if c == ".":
                g = self.valid_guess[idx]
                setc = set(g)
                self.position_could_be[idx] ^= setc
                self.log.debug(f"position {idx}: not '{g}'")
                if g not in self.in_expr:
                    self.log.debug(f"'{g}' not in expression")
                    self.not_in_expr |= setc

    def generate_legal_expressions(self):
        """
        If we did not have an expression file to load, generate legal
        equations.  This takes a while to run.
        """
        eqn: dict[str, bool] = {}
        e_l = self.expression_length
        equals_position = [e_l - 3, e_l - 2]  # Two-digit answers, then one.
        if e_l > 6:
            for i in range(e_l - 3, 3, -1):  # Then longer answers
                equals_position.append(i)
        # '=' cannot be farther to the left than the fourth character, because
        # the first three (at least) must be a OPR b .  Since the string length
        # is even, both sides cannot just be numbers, and the right hand side
        # has to be a non-negative integer without a leading zero (unless it
        # is just zero), so the equal sign can't be at the end.
        #
        # This is dumb, but what we are going to do is brute-force the solution
        # space, with the equals sign in the above place in each place in the
        # sequence based on my intuition that that the given sequence
        # represents the sequence of most likely places for it.
        #
        for eqp in equals_position:
            for exp_tuple in self.generate_expressions(eqp):
                q = "".join(exp_tuple)
                try:
                    _ = int(q)
                    continue
                    # It's an integer constant.  It evaluates to itself,
                    #  and it is not worth storing.
                except ValueError:
                    pass
                try:
                    lhs = self.parse_expr(q)
                    eqn = f"{q}={lhs}"
                    self.store_expr(q, lhs)
                except ValueError as exc:
                    self.log.debug(f"{q} does not parse: {exc}")
                    self.store_expr(q, None)
                    continue
                # Mark the equation as true
                self.log.debug(f"Storing '{eqn}'")
                self.store_expr(eqn, lhs)
                # Well, it's true, buuuuut...not valid by our rules.
                # So we don't store it as a valid equation.
                if len(eqn) == self.expression_length:
                    self.log.debug(f"Storing '{eqn}' as valid")
                    self._valid_equations[eqn] = lhs
                # I thought about storing all the equations that evaluated
                # to invalid answers, but it takes a lot of memory for
                # not much gain.

    def store_expr(self, expr: str, val: Optional[int]):
        """
        Determining whether an expression has a legal evaluation is
        expensive, so we build a cache so we only evaluate each expression
        once.
        """
        if expr in self._expr_by_str:
            oldval = self._expr_by_str[expr]
            if oldval == val:
                return
            raise ValueError(f"Does '{expr}' evaluate to {oldval} or {val}?")
        try:
            # There's no point in storing integer constants: testing equality
            # is faster than looking up the map and then testing equality.
            _ = int(expr)
            return
        except ValueError:
            pass
        self._expr_by_str[expr] = val
        if val is None:
            return
        self.log.debug(f"Stored '{expr}' -> {val}")
        if val not in self._expr_by_val:
            self._expr_by_val[val] = []
        if expr not in self._expr_by_val[val]:
            self._expr_by_val[val].append(expr)

    def generate_expressions(self, e_l: int) -> Iterator[Tuple[Any, ...]]:
        """
        Generate all expressions of length e_l.  Returns an iterator so we
        can chew through, and cache, all the ones that parse to an integer
        value.
        """
        legal_rhs_chars = set("=") ^ set(self.legal_chars)
        digits = set("+-*/") ^ set(legal_rhs_chars)
        assert e_l > 2, "expression length must be at least 3"
        assert (
            e_l < self.expression_length - 1
        ), f"expression length must be at most {self.expression_length - 2}"
        # We know the first and last character are digits
        exp_args = [digits]
        for _ in range(e_l - 2):
            exp_args.append(legal_rhs_chars)
        exp_args.append(digits)
        expr = product(*exp_args)  # itertools is awesome
        return expr

    def check_is_valid_number(self, n: str) -> int:
        """
        Check whether a string is a valid-by-Nerdle-rules number: return
        the corresponding int if so.
        """
        if not n:
            raise ValueError("The empty string is not a number")
        for c in n:
            if not c.isdigit():
                raise ValueError("numbers are made of digits")
        if len(n) > 1:
            if n[0] == "0":
                raise ValueError(
                    "Leading zeroes on multi-digit numbers are not allowed"
                )
        i_n = int(n)
        return i_n

    def restrict_possibilities(self) -> None:
        """
        Iterate through our remaining valid equations, eliminating the ones
        that don't fit the observed facts.
        """
        remainder = []
        rl = len(self.remaining_possibilities)
        for s in self.remaining_possibilities:
            try:
                self.validate_guess(s)
                remainder.append(s)
            except ValueError as exc:
                self.log.debug(f"'{s}' is eliminated: '{exc}'")
        rr = len(remainder)
        if rr == 0:
            raise OutOfEquations("No possible valid equations remain")
        self.log.debug(f"{rl - rr} equations eliminated: {rr} remain")
        self.remaining_possibilities = remainder

    def sort_remainder(self) -> None:
        """
        Return the "best" remaining possibilities, for some metric of "best"
        """
        # No idea what the best strategy here is.  Let's pick the ones with
        # the most unconfirmed characters?  (Eliminated characters were
        # eliminated from remaining_possibilities already)
        self.remaining_possibilities.sort(
            key=self.most_unused,
            reverse=True,
        )

    def most_unused(self, v: str) -> int:
        """
        Convenience sort method; most characters we don't know whether they're
        in the answer, ties broken by number of different characters.
        """
        cset = set(v)
        unknown = cset - self.in_expr
        score = self.expression_length * len(unknown) + len(cset)
        return score

    def choose_or_show_next_guess(self) -> None:
        """
        We have a sorted list of remaining guesses.  If we know the answer,
        pick the top one.  If we don't, display some to the user to prompt
        the next guess.
        """
        if self.answer:
            self.log.debug("Choosing best guess")
            self.guess = self.remaining_possibilities[0]
            self.valid_guess = self.guess
            return
        else:
            best = self.remaining_possibilities[: self.top]
            print(f"Best remaining possibilities: {', '.join(best)}")
