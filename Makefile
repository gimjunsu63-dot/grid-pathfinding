# Run inside the container (docker compose exec lab bash) or any Python 3.11+.
#
#   make run     demo: BFS / Dijkstra / A* on small hand-made maps
#   make test    unit tests
#   make clean   remove caches

.PHONY: all run test clean

all: test

run:
	@python3 src/main.py

test:
	@python3 -m unittest discover -s tests -v

clean:
	rm -rf src/__pycache__ tests/__pycache__
