FROM python:3.12-alpine
# The token lives only in this image, never in the agent's.
RUN mkdir -p /www && printf 'sidecar-7f3a' > /www/token
CMD ["python3", "-m", "http.server", "8000", "--directory", "/www"]
