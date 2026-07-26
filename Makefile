# Reproducible developer + CI gate entry points. `make gates` runs everything
# CI runs, locally, in order. The Rust toolchain is pinned by rust-toolchain.toml
# and the Python env is created with --system-site-packages so a distro pytorch
# is reused when present (matching CI).

PYTHON ?= python
VENV   ?= .venv
VPY     = $(VENV)/bin/python
RT      = rust/idr-intelligence-rt/Cargo.toml
ATTACK_BUNDLE ?= /home/lee/Desktop/cti/enterprise-attack/enterprise-attack.json

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VPY): ## Create the virtualenv and install the project + dev/export extras
	$(PYTHON) -m venv --system-site-packages $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -e '.[dev,export]'

.PHONY: venv
venv: $(VPY) ## Alias for the venv target

.PHONY: test
test: ## Run the Python test suite
	$(VPY) -m pytest -q

.PHONY: lint
lint: ## Ruff lint
	$(VPY) -m ruff check src tests scripts

.PHONY: types
types: ## mypy type check
	$(VPY) -m mypy src

.PHONY: bench
bench: ## Frozen benchmark floors (exit 1 on regression)
	cd "$$(mktemp -d)" && "$(CURDIR)/$(VPY)" -m idr_intelligence.cli benchmark --manifest "$(CURDIR)/benchmarks/v1.json"

.PHONY: rust
rust: ## Rust bridge: fmt check, clippy (-D warnings), tests
	cargo fmt --check --manifest-path $(RT)
	cargo clippy --all-targets --manifest-path $(RT) -- -D warnings
	cargo test --manifest-path $(RT)

.PHONY: gates
gates: test lint types bench rust ## Run every CI gate locally

.PHONY: attack-reference
attack-reference: ## Regenerate the MITRE ATT&CK reference (needs the STIX bundle)
	IDR_ATTACK_BUNDLE="$(ATTACK_BUNDLE)" $(VPY) scripts/ground_attack_reference.py

.PHONY: docker
docker: ## Build the Python engine image
	docker build -t idr-intelligence:local .

.PHONY: docker-rt
docker-rt: ## Build the minimal Rust serving image
	docker build -t idr-intelligence-rt:local rust/idr-intelligence-rt

.PHONY: clean
clean: ## Remove build/test artifacts (keeps the venv)
	rm -rf artifacts reports/demo.json .mypy_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	cargo clean --manifest-path $(RT) 2>/dev/null || true
