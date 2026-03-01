# Stoolap Makefile
# Build, test, lint, and benchmark targets

.PHONY: all build build-release build-tikv test test-all test-tikv test-tikv-complex \
        bench bench-tikv lint fmt clippy clean doc \
        tikv-up tikv-down tikv-logs tikv-clean

# ─── Configuration ──────────────────────────────────────────────────

CARGO        := cargo
TIKV_COMPOSE := docker compose -f docker-compose.tikv.yml
TIKV_PD      ?= 127.0.0.1:2379

# ─── Build ──────────────────────────────────────────────────────────

all: lint test build

## Build debug binary (default features: cli, parallel)
build:
	$(CARGO) build --features cli

## Build optimized release binary
build-release:
	$(CARGO) build --release --features cli

## Build with TiKV support
build-tikv:
	$(CARGO) build --features "cli,tikv"

## Build release with TiKV support
build-tikv-release:
	$(CARGO) build --release --features "cli,tikv"

# ─── Test ───────────────────────────────────────────────────────────

## Run unit and integration tests (default features)
test:
	$(CARGO) test

## Run all tests including TiKV compilation check (no TiKV cluster needed)
test-all:
	$(CARGO) test --features tikv

## Run library unit tests only
test-lib:
	$(CARGO) test --lib

## Run TiKV integration tests (requires running TiKV cluster)
## Start cluster first: make tikv-up
test-tikv: tikv-check
	TIKV_PD_ENDPOINTS=$(TIKV_PD) $(CARGO) test --features tikv --test tikv_test -- --test-threads=1

## Run complex TiKV integration tests (joins, subqueries, concurrent txns)
test-tikv-complex: tikv-check
	TIKV_PD_ENDPOINTS=$(TIKV_PD) $(CARGO) test --features tikv --test tikv_complex_test -- --test-threads=1

## Run full TiKV test suite (basic + complex)
test-tikv-all: tikv-check
	TIKV_PD_ENDPOINTS=$(TIKV_PD) $(CARGO) test --features tikv --test tikv_test --test tikv_complex_test -- --test-threads=1

## Check that TiKV is reachable (used as dependency for tikv test targets)
tikv-check:
	@curl -sf http://$(TIKV_PD)/pd/api/v1/members > /dev/null 2>&1 || \
		(echo "Error: TiKV cluster not reachable at $(TIKV_PD)"; \
		 echo "Start it with: make tikv-up"; exit 1)

# ─── Benchmarks ─────────────────────────────────────────────────────

## Run all benchmarks (MVCC in-memory)
bench:
	$(CARGO) bench

## Run benchmarks with SQLite comparison
bench-sqlite:
	$(CARGO) bench --features sqlite

## Run benchmarks with DuckDB comparison
bench-duckdb:
	$(CARGO) bench --features duckdb

## Run TiKV vs MVCC comparison benchmark (requires running TiKV cluster)
bench-tikv: tikv-check
	TIKV_PD_ENDPOINTS=$(TIKV_PD) $(CARGO) bench --features tikv --bench tikv_comparison

# ─── Lint ───────────────────────────────────────────────────────────

## Run all lints (format check + clippy)
lint: fmt-check clippy

## Check code formatting
fmt-check:
	$(CARGO) fmt --all -- --check

## Auto-format code
fmt:
	$(CARGO) fmt --all

## Run clippy linter
clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

# ─── Documentation ──────────────────────────────────────────────────

## Generate Rust documentation
doc:
	$(CARGO) doc --no-deps --features tikv

## Generate and open Rust documentation
doc-open:
	$(CARGO) doc --no-deps --features tikv --open

# ─── TiKV Cluster Management ───────────────────────────────────────

## Start local TiKV cluster (PD + TiKV, host networking)
tikv-up:
	$(TIKV_COMPOSE) up -d
	@echo "Waiting for TiKV cluster to bootstrap..."
	@for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		if curl -sf http://$(TIKV_PD)/pd/api/v1/members > /dev/null 2>&1; then \
			echo "TiKV cluster is ready at $(TIKV_PD)"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Warning: TiKV cluster may not be fully ready yet"

## Stop local TiKV cluster
tikv-down:
	$(TIKV_COMPOSE) down

## View TiKV cluster logs
tikv-logs:
	$(TIKV_COMPOSE) logs -f

## Stop TiKV cluster and delete all data
tikv-clean:
	$(TIKV_COMPOSE) down -v

# ─── CI ─────────────────────────────────────────────────────────────

## Run the full CI pipeline locally
ci: lint test-lib test
	@echo "CI checks passed"

## Run CI pipeline including TiKV tests (requires running TiKV cluster)
ci-tikv: lint test-lib test test-tikv
	@echo "CI checks (with TiKV) passed"

# ─── Utilities ──────────────────────────────────────────────────────

## Clean build artifacts
clean:
	$(CARGO) clean

## Print lines of code
loc:
	@find src -name '*.rs' | xargs wc -l | tail -1

## Run the Stoolap CLI
run:
	$(CARGO) run --features cli --release

# ─── Help ───────────────────────────────────────────────────────────

## Show this help
help:
	@echo "Stoolap Makefile Targets"
	@echo ""
	@echo "Build:"
	@echo "  make build              Build debug binary"
	@echo "  make build-release      Build optimized release binary"
	@echo "  make build-tikv         Build with TiKV support"
	@echo ""
	@echo "Test:"
	@echo "  make test               Run unit + integration tests"
	@echo "  make test-lib           Run library unit tests only"
	@echo "  make test-tikv          Run TiKV integration tests"
	@echo "  make test-tikv-complex  Run complex TiKV integration tests"
	@echo "  make test-tikv-all      Run all TiKV tests"
	@echo ""
	@echo "Benchmark:"
	@echo "  make bench              Run benchmarks (MVCC)"
	@echo "  make bench-sqlite       Run benchmarks vs SQLite"
	@echo "  make bench-tikv         Run MVCC vs TiKV comparison"
	@echo ""
	@echo "Lint:"
	@echo "  make lint               Run format check + clippy"
	@echo "  make fmt                Auto-format code"
	@echo ""
	@echo "TiKV Cluster:"
	@echo "  make tikv-up            Start local TiKV cluster"
	@echo "  make tikv-down          Stop TiKV cluster"
	@echo "  make tikv-clean         Stop + delete all TiKV data"
	@echo "  make tikv-logs          View TiKV cluster logs"
	@echo ""
	@echo "Other:"
	@echo "  make doc                Generate documentation"
	@echo "  make ci                 Run full CI pipeline locally"
	@echo "  make clean              Clean build artifacts"
	@echo "  make run                Run Stoolap CLI (release)"
