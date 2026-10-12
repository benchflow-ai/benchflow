#!/bin/bash
python3 -c "import urllib.request; open('/app/token.txt', 'w').write(urllib.request.urlopen('http://api:8000/token', timeout=10).read().decode())"
