# Nerdle Solver

Nerdle's (https://nerdlegame.com) is like Wordle
(https://www.powerlanguage.co.uk/wordle/), except that the "word" you're
entering is an eight- (or six-) character arithmetic expression.

## Interactive use

If you're using it interactively--that is, without specifying the
`--answer` option, input the results you get as an
eight-(resp. six-)character string, where `.` means it's not in the
word, `?` means it's in the word but in the wrong place, and `!` means
it's in the right place.  In the Nerdle UI, `!` is green, and `?` is
purple.

So: if you are being a horrible person and using this to cheat at
Nerdle, what you do is put your initial guess into both this program and
Nerdle.  Then you take Nerdle's answer, transcribe it into a result this
program will accept (that is, character string of the right length
consisting of only `!`, `?`, and `.`), put that into the program, and
then type the next suggestion it gives you into Nerdle.

## Non-interactive

If you know the answer and want to see if this program will solve it
faster than you did, try `--answer` and see how many guesses it takes.

## Options

You can set the initial guess with `--initial-guess`.

If you set `--top` you will get more or fewer displayed best guesses
than the default of 5.

You can let it run for more guesses with `--guesses` (the default is 6,
per Wordle) and you can play with different expression lengths with
`--length` (the default is 8, and Nerdle's "easy mode" is 6).

`--debug` will emit copious output about what the solver is doing as it
does it.

## Generating equation files

Before first use, you may want to run `generate-data`, which will put
files in the `static` directory below wherever the `nerdle_solver.py`
file exists.

If you don't do that, or if that isn't writeable, the solver will take a
long time to start up, because it is generating the list of all
valid-by-Nerdle-rules eight- (resp. six-) letter equations.

These files are actually fairly small: there just aren't that many
equations that satisfy all the Nerdle rules.

```
adam@m1-wired:~/git/nerdle-solver$ ls -l src/nerdle_solver/static
total 2000
-rw-r--r--  1 adam  staff     6291 Feb  7 22:48 equations-6.json
-rw-r--r--  1 adam  staff  1014184 Feb  7 22:48 equations-8.json
```

## Developing

If you want to play around with this yourself, ``make init`` will set up
the pre-commit hooks for you.  The solver itself has no external
requirements: everything in it is in the Python 3.8 standard library;
however, the test suite and pre-commit hooks have some external packages
they require.
