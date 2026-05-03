.PHONY: lint type-check check

# Runs just the linter
lint:
	uvx ruff check .

# Runs just the type checker
type-check:
	npx pyright

# Runs everything in sequence
check: lint type-check
	python -m compileall -q .
	@echo "All 'compilation' checks passed!"