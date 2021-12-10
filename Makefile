all: base generation

base:
	@echo "Checking to make sure submodules are cloned..."
	git submodule update --init --recursive
	@echo "Installing: local python libraries..."
	pip install -e lib/*
	pip install -e .
	@echo "Completed: local python library installation"

generation:
	@echo "Compiling: Kronecker generation from snap..."
	cd libc/snap/examples/krongen/; make all
	@echo "Completed: Kronecker generation compilation"
	@echo "Installing: graph-tools..."
	conda install -c conda-forge graph-tool
	@echo "Completed: graph-tools installation"
