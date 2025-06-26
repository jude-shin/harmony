#!/bin/bash

# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_003.png" \
#      -F "product_line_string=pokemon" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_002.png" \
#      -F "product_line_string=pokemon" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_001.png" \
#      -F "product_line_string=pokemon" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_004.png" \
#      -F "product_line_string=pokemon" \
#      https://ai.storepass.co/predict
# 
# 
# curl -X POST \
#      -F "image=@$HOME/harmony/testing/Selection_005.png" \
#      -F "product_line_string=pokemon" \
#      https://ai.storepass.co/predict
#

curl -X POST \
     -F "image=@$HOME/harmony/testing/Selection_001.png" \
     -F "product_line_string=pokemon" \
		 -w "\\nResponse Time: %{time_total} seconds\\n" \
     https://ai.storepass.co/predict


