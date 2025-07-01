#!/bin/bash

# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_003.png" \
#      -F "product_line_string=pokemon" \
# 		 -F "threshold=90" \
# 		 -w "\\nResponse Time: %{time_total} seconds\\n" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_002.png" \
#      -F "product_line_string=pokemon" \
# 		 -F "threshold=90" \
# 		 -w "\\nResponse Time: %{time_total} seconds\\n" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_001.png" \
#      -F "product_line_string=pokemon" \
# 		 -F "threshold=90" \
# 		 -w "\\nResponse Time: %{time_total} seconds\\n" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_004.png" \
#      -F "product_line_string=pokemon" \
# 		 -F "threshold=90" \
# 		 -w "\\nResponse Time: %{time_total} seconds\\n" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_005.png" \
#      -F "product_line_string=pokemon" \
# 		 -F "threshold=90" \
# 		 -w "\\nResponse Time: %{time_total} seconds\\n" \
#      https://ai.storepass.co/predict

curl -X POST \
	-F "images=@$HOME/harmony/testing/Selection_001.png" \
	-F "images=@$HOME/harmony/testing/Selection_002.png" \
	-F "images=@$HOME/harmony/testing/Selection_003.png" \
	-F "images=@$HOME/harmony/testing/Selection_004.png" \
	-F "images=@$HOME/harmony/testing/Selection_005.png" \
	-F "product_line_string=pokemon" \
	-F "threshold=90" \
	-w "\\nResponse Time: %{time_total} seconds\\n" \
	https://ai.storepass.co/predict

# curl -X POST \
#      -F "image=@$HOME/harmony/testing/1.png" \
#      -F "product_line_string=lorcana" \
# 		 -F "threshold=90" \
# 		 -w "\\nResponse Time: %{time_total} seconds\\n" \
#      https://ai.storepass.co/predict


