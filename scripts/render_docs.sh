#!/bin/bash

sphinx-apidoc -f -P -o docs/ $1
# sphinx-apidoc -F -f -P -o docs/ $1 # for initial docs
cd docs
make html
