FROM ubuntu:24.04

COPY --chmod=755 sv-grade /usr/bin/sv-grade
# Only in the verifier image: /out is a relative link to usr/local, so a
# packed out/bin/x would land in /usr/local/bin, ahead of /usr/bin on PATH.
# (GNU tar 1.35 refuses to follow an absolute link here; busybox tar follows
# both.)
RUN ln -s usr/local /out
COPY --chmod=755 test.sh /tests/test.sh
