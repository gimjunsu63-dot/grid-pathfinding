# Run inside the container (docker compose exec lab bash) or any Python 3.11+.
#
#   make run          demo: BFS / Dijkstra / A* on small hand-made maps
#   make maps         print a small sample of every map kind
#   make test         unit tests
#   make bench        all experiments -> results/*.csv (a few minutes)
#   make bench-quick  tiny version of every experiment -> results/quick/
#   make figures      pictures of what each algorithm explored -> figures/maps/
#   make analyze      charts + tables from results/*.csv -> figures/charts/, figures/RESULTS.md
#   make summary      one-page summary of all results (terminal + figures/SUMMARY.md)
#   make results      bench + analyze + summary + figures (everything, about 7 minutes)
#   make clean        remove caches

.PHONY: all run maps test bench bench-quick figures analyze summary results clean

all: test

run:
	@python3 src/main.py

maps:
	@python3 src/maps.py

test:
	@python3 -m unittest discover -s tests -v

bench:
	@python3 src/bench.py

bench-quick:
	@python3 src/bench.py --quick

figures:
	@python3 src/visualize.py

analyze:
	@python3 src/analyze.py

summary:
	@python3 src/summary.py --write

results: bench analyze summary figures

clean:
	rm -rf src/__pycache__ tests/__pycache__ results/quick
