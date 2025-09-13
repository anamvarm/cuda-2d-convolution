#!/bin/bash

# validate.sh - A script to validate the correctness of the CUDA convolution implementation
# by comparing its output to a reference CPU implementation.

echo "======================================="
echo "    CUDA Convolution Validation"
echo "======================================="
echo

# Set error handling: exit script on any failure
set -e

# Step 1: Build both programs using the Makefile
echo "1. Building GPU and CPU executables..."
make clean > /dev/null  # Clean previous builds, silence output
make all > /dev/null    # Build everything, silence output
echo "   Build successful."
echo

# Step 2: Run both programs and capture their output
echo "2. Executing programs..."
./convolution_gpu > gpu_output.txt
./convolution_cpu > cpu_output.txt
echo "   Execution complete. Outputs saved to gpu_output.txt and cpu_output.txt."
echo

# Step 3: Extract only the result matrices from the output files.
# This ignores the printed input/mask matrices and headers.
# We use 'tail' to get the last N lines (the result matrix).
echo "3. Extracting results for comparison..."
MATRIX_SIZE=8 # Set this to the size of your output matrix

# Get the last $MATRIX_SIZE lines from each file, then flatten into a single line for comparison
tail -n $MATRIX_SIZE gpu_output.txt | tr -s ' ' | tr '\n' ' ' > gpu_results_flat.txt
tail -n $MATRIX_SIZE cpu_output.txt | tr -s ' ' | tr '\n' ' ' > cpu_results_flat.txt
echo "   Results extracted."
echo

# Step 4: Compare the two results
echo "4. Comparing results..."
echo
echo "   GPU Results:"
cat gpu_results_flat.txt
echo
echo
echo "   CPU Results:"
cat cpu_results_flat.txt
echo
echo

# Use `diff` to check if the files are identical. Silence its output.
if diff -q gpu_results_flat.txt cpu_results_flat.txt > /dev/null; then
    echo "   ✅ SUCCESS: GPU and CPU results are identical!"
    VALIDATION_RESULT="PASS"
else
    echo "   ❌ FAILURE: Results do not match!"
    echo "   Here is the difference:"
    diff -y gpu_results_flat.txt cpu_results_flat.txt || true # `|| true` prevents script from exiting due to diff's non-zero exit code
    VALIDATION_RESULT="FAIL"
fi
echo

# Step 5: (Optional) Numeric comparison with tolerance using a Python script
echo "5. Numeric comparison with tolerance..."
# Create a small Python script to compare floats accurately
python3 - <<EOF > tolerance_result.txt
import numpy as np

# Load the data, converting strings to floats
gpu_data = np.loadtxt('gpu_results_flat.txt')
cpu_data = np.loadtxt('cpu_results_flat.txt')

# Check if all elements are close within a tolerance
tolerance = 1e-5
is_close = np.allclose(gpu_data, cpu_data, atol=tolerance)

if is_close:
    print("✅ SUCCESS: Results are numerically identical within tolerance of {:.0e}".format(tolerance))
else:
    # Find and print the maximum error
    max_error = np.max(np.abs(gpu_data - cpu_data))
    print("❌ FAILURE: Results differ. Maximum error is: {:.6f}".format(max_error))
EOF

cat tolerance_result.txt
echo

# Step 6: Cleanup and final message
echo "6. Cleaning up temporary files..."
rm -f gpu_output.txt cpu_output.txt gpu_results_flat.txt cpu_results_flat.txt tolerance_result.txt
echo
echo "======================================="
echo "    Validation Result: $VALIDATION_RESULT"
echo "======================================="

# Exit with appropriate code (0 for success, 1 for failure)
if [ "$VALIDATION_RESULT" = "PASS" ]; then
    exit 0
else
    exit 1
fi
