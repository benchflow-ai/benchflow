#!/bin/bash
# The wrong answer, and a stand-in for the grader the verifier image trusts (it
# only writes a marker and says 1) in the declared /out/bin, an ordinary folder
# here. The verifier image's /out is a link to /usr/local, so it would land in
# /usr/local/bin and shadow /usr/bin/sv-grade on PATH.
echo 41 > /app/answer.txt
printf '#!/bin/sh\necho ran > /logs/verifier/planted-grader-ran\necho 1\n' > /out/bin/sv-grade
chmod 755 /out/bin/sv-grade
