#!/bin/bash
# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
echo "Usage: $0 path_to_zip_file"
exit 1
fi
ZIP_FILE="$1"
EXPECTED_FIRST_LINE="Total density after 100 timesteps: 81981.3"
# Check if the input file is a zip file
if [[ "${ZIP_FILE}" != *.zip ]]; then
echo "Error: The provided file is not a zip file."
exit 1
fi
# Create a temporary directory for unzipping
TEMP_DIR="tmp"
echo "Unzipping file to $TEMP_DIR"
# Unzip the provided zip file
unzip "${ZIP_FILE}" -d "${TEMP_DIR}" 2>/dev/null
# Check if unzip was successful
if [ $? -ne 0 ]; then
echo "Error: Failed to unzip the file."
rm -rf "${TEMP_DIR}"
exit 1
fi
# Find directories in the unzipped folder
for entry in "$TEMP_DIR"/*
do
if [ -d "$entry" ]; then
# Recursively search for Makefile or makefile in the directory
makefile_path=$(find "$entry" -type f \( -name "Makefile" -o -name "makefile" \) | head -n 1)
# If a Makefile is found
if [ -n "$makefile_path" ]; then
echo "Found Makefile in $entry"
# Move to the directory containing the Makefile
makefile_dir=$(dirname "$makefile_path")
cd "$makefile_dir"
# Run make
make
if [ $? -ne 0 ]; then
echo "Error: Make failed in $makefile_dir"
cd - > /dev/null
continue
fi
# Check if the executable fluid_sim is present
if test -f "./fluid_sim"; then
echo "$entry - exe is ok"
# Run perf stat on the executable
srun --partition cpar perf stat -e cycles ./fluid_sim > run1.txt 2>&1
# Validate the first line and specific value in the output
FIRST_LINE=$(head -n 1 run1.txt)
if [[ "$FIRST_LINE" == "$EXPECTED_FIRST_LINE" ]] && grep -q "$EXPECTED_VALUE" run1.txt; then
echo "Output validation passed"
else
echo "Output validation failed:"
echo "Expected first line: '$EXPECTED_FIRST_LINE'"
echo "Actual first line: '$FIRST_LINE'"
fi
# Output the entire contents of run1.txt
echo "Contents of run1.txt:"
cat run1.txt
else
echo "fluid_sim not found in $makefile_dir"
fi
# Return to the original directory
cd - > /dev/null
else
echo "No Makefile found in $entry, skipping."
fi
fi
done
# Clean up the temporary directory
rm -rf "${TEMP_DIR}"