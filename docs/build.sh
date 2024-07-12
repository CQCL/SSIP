#! /bin/bash

mkdir build

touch build/.nojekyll  # Disable jekyll to keep files starting with underscores

cp ./_static/_redirect.html ./build/index.html

sphinx-build -b html ./api-docs ./build/api-docs
