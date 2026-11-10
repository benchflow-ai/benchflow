# The PDF reviewer image: BenchFlow's reviewer image (REVIEWER_IMAGE in
# src/benchflow/review/options.py) plus poppler-utils (pdftotext, pdftoppm,
# pdfinfo) and the reviewer's packages with PyMuPDF, at the pins
# src/benchflow/review/wrapper.py installs (REVIEWER_ARTIFACT_PACKAGES and
# REVIEWER_PDF_PACKAGES). BenchFlow installs nothing into it. Build it under
# the tag REVIEWER_PDF_IMAGE names, the first 12 hex of this file's sha256:
#
#   tag=benchflow-reviewer-pdf:$(sha256sum docker/reviewer-pdf/Dockerfile | cut -c1-12)
#   docker build -t "$tag" - < docker/reviewer-pdf/Dockerfile
#
# and pass --reviewer-image "$tag" with the Docker reviewer sandbox. A backend
# that pulls from a registry (Daytona) cannot start a local image; give it
# --reviewer-pdf-tools on the default image instead.
FROM python@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --root-user-action=ignore \
      numpy==2.2.6 pypdf==5.9.0 PyMuPDF==2.1
