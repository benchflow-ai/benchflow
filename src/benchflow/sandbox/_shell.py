"""Dependency-free constants for sandbox command bootstrap."""

# Never look up a trust probe's tools in the image's PATH extras.
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"

# The folders the verifier's PATH always has (``lockdown.VERIFIER_ENV``), and
# the PATH root's commands use while a trial is verified.
VERIFIER_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
