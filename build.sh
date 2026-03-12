#!/bin/bash
set -e
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Copying Plotly JS bundle from installed package..."
python3 -c "
import plotly, os, shutil, glob
pkg = os.path.dirname(plotly.__file__)
# Find the minified bundle
candidates = glob.glob(pkg + '/**/plotly.min.js', recursive=True)
if not candidates:
    candidates = glob.glob(pkg + '/**/*.min.js', recursive=True)
if candidates:
    # Pick smallest file (basic bundle)
    src = sorted(candidates, key=os.path.getsize)[0]
    print(f'Found: {src} ({os.path.getsize(src)//1024}KB)')
    shutil.copy(src, 'static/plotly.min.js')
    print('Copied to static/plotly.min.js')
else:
    print('No JS bundle found in plotly package')
    print('Files in package:', os.listdir(pkg))
"

echo "Build complete."
