#!/bin/bash
python3 -c "print(sum(all(n % d for d in range(2, int(n**0.5) + 1)) for n in range(2, 50000)))" > /app/answer.txt
