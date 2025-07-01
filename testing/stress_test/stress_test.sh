#!/bin/bash

# === Configuration ===
IMG_DIR="$HOME/harmony/testing"
PRODUCT_LINE="pokemon"
THRESHOLD=90
ENDPOINT="https://ai.storepass.co/predict"
LOG_FILE="curl_test_log.csv"
REPEATS=4  # number of curls per batch size

# Clear and initialize the log file
echo "num_images,response_time_sec" > "$LOG_FILE"

# Gather all image paths
all_images=($(find "$IMG_DIR" -type f -name "*.png" | sort))
total_images=${#all_images[@]}

echo "Found $total_images images."

# Generate Fibonacci-based batch sizes up to total_images
fib=(1 2)
while true; do
	next=$((fib[-1] + fib[-2]))
	if [ "$next" -ge "$total_images" ]; then
		fib+=("$total_images")
		break
	fi
	fib+=("$next")
done

# Loop through each batch size
for batch_size in "${fib[@]}"; do
	echo "Testing batch size: $batch_size"

	for run in $(seq 1 $REPEATS); do
		# Build curl command
		curl_cmd=(curl -s -w "\n%{time_total}" -o /dev/null -X POST)

		for ((i=0; i<batch_size; i++)); do
			img_path="${all_images[i]}"
			curl_cmd+=(-F "images=@$img_path")
		done

		curl_cmd+=( -F "product_line_string=$PRODUCT_LINE" -F "threshold=$THRESHOLD" "$ENDPOINT")

				# Run the curl and extract response time
				result="$("${curl_cmd[@]}")"
				time_taken=$(echo "$result" | tail -n 1)

				# Log to CSV
				echo "$batch_size,$time_taken" >> "$LOG_FILE"
				echo "  Run $run: ${time_taken}s"
			done
		done

		echo "Done. Results written to $LOG_FILE"

