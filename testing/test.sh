#!/bin/bash
curl -X POST \
	-F "images=@$HOME/harmony/testing/Selection_001.png" \
	-F "images=@$HOME/harmony/testing/Selection_002.png" \
	-F "images=@$HOME/harmony/testing/Selection_003.png" \
	-F "images=@$HOME/harmony/testing/Selection_004.png" \
	-F "images=@$HOME/harmony/testing/Selection_005.png" \
	-F "images=@$HOME/harmony/testing/Selection_006.png" \
	-F "images=@$HOME/harmony/testing/Selection_007.png" \
	-F "images=@$HOME/harmony/testing/Selection_008.png" \
	-F "images=@$HOME/harmony/testing/Selection_009.png" \
	-F "images=@$HOME/harmony/testing/Selection_010.png" \
	-F "images=@$HOME/harmony/testing/Selection_011.png" \
	-F "images=@$HOME/harmony/testing/Selection_012.png" \
	-F "product_line_string=pokemon" \
	-F "threshold=90" \
	-w "\\nResponse Time: %{time_total} seconds\\n" \
	https://ai.storepass.co/predict
