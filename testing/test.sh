#!/bin/bash

curl -X POST \
     -F "image=@$HOME/harmony/testing/Selection_003.png" \
     -F "product_line_string=pokemon" \
     https://ai.storepass.co/predict


