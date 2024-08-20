# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../build

# Help target: Displays available Sphinx build targets
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help clean

# Clean target: Removes the build directory if it exists
clean:
	@if [ -d "$(BUILDDIR)" ]; then rm -rf "$(BUILDDIR)"; fi

# Explicit rule for Makefile so it's not caught by the % wildcard
Makefile: ;

# Catch-all target: Routes all unknown targets to Sphinx using the Make mode
%: clean
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
