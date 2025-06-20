curl -X POST \
     -F "image=@$HOME/harmony/testing/1.png" \
     -F "threshold=0.6" \
     -F "img_width=313" \
     -F "img_height=437" \
     http://localhost:8000/predict/pokemon

