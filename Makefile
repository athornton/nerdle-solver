all: wheel

.PHONY: init
init: .git/hooks/pre-commit
	pip install --editable .
	pip install --upgrade pip-tools 'pip<22' setuptools
	rm -rf .tox
	pip install --upgrade tox pre-commit pytest
	pre-commit install

DATADIR=src/nerdle_solver/static/
testdata=$(DATADIR)expressions-6.json \
         $(DATADIR)expressions-8.json

$(testdata):
	generate-data

testdata: $(testdata)

.PHONY: test
test: $(testdata)
	pytest tests

wheel:
	pip wheel -e .

cleanup=tests/__pycache__                         \
        scripts/__pycache__                       \
        src/nerdle_solver/__pycache__             \
        src/nerdle_solver/nerdle_solver.egg_info  \
        build                                     \
        *.whl

.PHONY: clean
clean:
	-rm -rf $(cleanup)

