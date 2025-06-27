# docker run -p 8501:8501 \
#   --mount type=bind,source=/path/to/my_model/,target=/models/my_model \
#   -e MODEL_NAME=my_model -t tensorflow/serving
# 


# docker run -p 8501:8501 \
# 	--mount type=bind,source=/home/jude/harmony/models/lorcana/,target=/models/harmony/lorcana \
# 	-e MODEL_NAME=/harmony/lorcana -t tensorflow/serving




docker run -it -v /home/jude/harmony/saved_models/lorcana:/models/lorcana -p 8605:8605 --entrypoint /bin/bash tensorflow/servin
