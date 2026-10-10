FROM --platform=linux/amd64 python:3.13 AS base
COPY tests /opt/tests
COPY solution.sh /opt/solution.sh
COPY data /srv/data

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
ENV MODE=test
WORKDIR /srv
EXPOSE 8080
ENTRYPOINT ["python", "-m", "http.server"]
