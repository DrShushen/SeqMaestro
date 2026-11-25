"""`python -m seqmaestro.cli`, the same command as `seqmaestro`, for a shell in which the
environment is not activated."""

from seqmaestro.cli import app

app(prog_name="seqmaestro")
