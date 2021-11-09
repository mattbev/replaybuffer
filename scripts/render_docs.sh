#!/bin/bash

sphinx-apidoc -f -P -o docs/ $1
cd docs
make html
