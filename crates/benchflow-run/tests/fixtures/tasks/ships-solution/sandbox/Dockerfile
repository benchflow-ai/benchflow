FROM ubuntu:24.04
WORKDIR /app
# Deliberately broken: the reference solution is baked into the image.
RUN mkdir -p /solution && printf '#!/bin/bash\necho terces > /app/out.txt\n' > /solution/solve.sh && chmod 755 /solution/solve.sh
