FROM ubuntu:24.04
# Deliberately broken: a file in /etc, a library folder, and /opt are writable by everyone.
RUN chmod 666 /etc/environment && chmod o+w /usr/lib && mkdir -p /opt/tools && chmod 777 /opt/tools
WORKDIR /app
