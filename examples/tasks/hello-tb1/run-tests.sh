#!/bin/bash
[ "$(cat /app/hello.txt 2>/dev/null | tr -d '\n')" = "Hello, world!" ]
