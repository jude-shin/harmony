curl -X POST \
     -F "image=@$HOME/harmony/testing/1.png" \
     -F "threshold=0.6" \
     -F "img_width=255" \
     -F "img_height=255" \
     http://localhost:8000/predict/pokemon

