## Set this to point to your cutlass directory, or set it as an environment
CUTLASS_DIR=${PWD}/../external/cutlass

default:
	CUTLASS_DIR=${CUTLASS_DIR} pip3 install --no-build-isolation cutlass_gemm/

clean:
	pip3 uninstall cutlass-gemm
