#!/bin/bash
# Not the tool: a folder at the declared path, which was absent in the clean image.
mkdir -p /usr/local/bin/sv-tool
echo 'not a tool' > /usr/local/bin/sv-tool/readme
