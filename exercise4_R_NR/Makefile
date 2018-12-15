TEST_CMD = python3 -m doctest -v
CHECKSTYLE_CMD = flake8

all: compile test checkstyle

compile:
	@echo "Nothing to compile for Python"

test:
	$(TEST_CMD) *.py -v

checkstyle:
	$(CHECKSTYLE_CMD) *.py

clean:
	rm -f *.pyc
	rm -rf __pycache__
	rm -f *.py~
	rm -rf ./dqn/__pycache__
	rm -f ./dqn/*.py~