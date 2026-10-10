FROM ubuntu:24.04
# Deliberately careless: the build folder holds a .netrc (a dummy, not a real login), and the image keeps two copies of it, one only root can read and one anyone can.
COPY .netrc /root/.netrc
RUN chmod 600 /root/.netrc && mkdir -p /opt/deploy
COPY .netrc /opt/deploy/.netrc
RUN chmod 644 /opt/deploy/.netrc
WORKDIR /app
