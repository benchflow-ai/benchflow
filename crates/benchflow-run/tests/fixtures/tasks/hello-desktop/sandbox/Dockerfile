# A minimal browser desktop for computer-use agents (computer-1): Xvfb, the
# xfwm4 window manager, Debian's Chromium, xdotool for input, and ImageMagick
# for screenshots. After the image Harbor's computer-1 calibration task uses.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        dbus-x11 \
        fonts-liberation \
        imagemagick \
        procps \
        x11-utils \
        xauth \
        xdotool \
        xfwm4 \
        xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && chmod 1777 /tmp /var/tmp

WORKDIR /app
COPY hello.html /app/hello.html
RUN chmod 644 /app/hello.html
