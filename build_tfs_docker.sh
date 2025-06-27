#!/bin/bash

docker run -it \
	--rm \
	-p 8605:8605 \
	-v "$(pwd)/saved_models/lorcana/:/models/lorcana/" tensorflow/serving \
	--model_config_file=/models/lorcana/tfs.config \
	--model_config_file_poll_wait_seconds=60 \
	--rest_api_port=8605




