"""Guards FF-sw-other IMAP sequence stability from 363d815b."""

import pytest

from benchflow.adapters._toolathlon_fake_mail import MailStore


@pytest.mark.parametrize(
    "seqs,remaining,expunged",
    [
        ([1, 2], [b"C"], [1, 1]),
        ([2, 3], [b"A"], [2, 2]),
        ([1, 3], [b"B"], [1, 2]),
        ([2, 2], [b"A", b"C"], [2]),
    ],
)
def test_deleted_flags_do_not_change_sequence_numbers(
    tmp_path, seqs, remaining, expunged
):
    store = MailStore(tmp_path)
    for raw in (b"A", b"B", b"C"):
        store.append("dummy", "INBOX", raw)
    for seq in seqs:
        store.mark_deleted("dummy", "INBOX", [seq])
        assert [p.read_bytes() for p in store.messages("dummy", "INBOX")] == [
            b"A",
            b"B",
            b"C",
        ]
    assert store.expunge("dummy", "INBOX") == expunged
    assert [p.read_bytes() for p in store.messages("dummy", "INBOX")] == remaining
