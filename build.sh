#!/bin/bash
pip install -r requirements.txt
echo "Downloading Plotly bundle..."
curl -sL https://cdn.plot.ly/plotly-basic-2.26.0.min.js -o static/plotly-basic.min.js
echo "Plotly downloaded: $(wc -c < static/plotly-basic.min.js) bytes"
