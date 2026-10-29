"""Egress URL policy: listed pages named inside other URLs.

Guards the denylist egress mode from PR #1113 against a reader, archive,
redirect or translation service handed a blocked address, which fetches the
page from a host the denylist does not name.
"""

from __future__ import annotations

import pytest

from benchflow.sandbox._egress_policy import Policy

PAPER = "https://arxiv.org/abs/2401.12345"


class TestEmbeddedUrls:
    @pytest.mark.parametrize(
        ("host", "target"),
        [
            ("r.jina.ai", "/https://arxiv.org/abs/2401.12345"),
            ("web.archive.org", "/web/2026id_/https://arxiv.org/abs/2401.12345v2"),
            ("web.archive.org", "/web/2026*/arxiv.org/abs/2401.12345"),
            (
                "translate.google.com",
                "/translate?u=https://www.arxiv.org/abs/2401.12345",
            ),
            ("redirect.test", "/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2401.12345&x=1"),
            (
                "redirect.test",
                "/go?to=https%253A%252F%252Farxiv.org%252Fabs%252F2401.12345",
            ),
            ("reader.test", "/https://arxiv.org:443/abs/../abs/2401.12345"),
            ("search.test", "/search?q=arxiv.org/ABS/2401.12345"),
            ("r.jina.ai", "/https://arxiv.org./abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv\u3002org/abs/2401.12345"),
            ("r.jina.ai", "/https://\uff41rxiv\uff0eorg/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv%EF%BC%8Eorg/abs/2401.12345"),
            ("r.jina.ai", "/https://ar%C2%ADxiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://ar%E2%80%8Bxiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv.o%E2%81%A0rg/abs/2401.12345"),
            ("r.jina.ai", "/https://ar%09xiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv.org/ab%0As/2401.12345"),
            ("r.jina.ai", "/https://arx%0Div.org/abs/2401.12345"),
            # An empty port is the default port to a URL parser.
            ("r.jina.ai", "/https://arxiv.org:/abs/2401.12345"),
            # Google Translate's site proxy writes the origin into its own name.
            ("arxiv-org.translate.goog", "/abs/2401.12345?_x_tr_sl=auto&_x_tr_tl=en"),
            ("www-arxiv-org.translate.goog", "/abs/2401.12345"),
        ],
    )
    def test_listed_page_inside_another_url_is_refused(self, host, target):
        rule = Policy([PAPER], []).url_rule(host, target)
        assert rule == "url:arxiv.org/abs/2401.12345"

    def test_translate_proxy_names_with_hyphens_decode(self):
        policy = Policy([], ["fp-viewer.vercel.app"])
        assert (
            policy.url_rule("fp--viewer-vercel-app.translate.goog", "/")
            == "host:fp-viewer.vercel.app"
        )

    def test_listed_host_inside_another_url_covers_subdomains(self):
        policy = Policy([], ["alphaxiv.org"])
        assert (
            policy.url_rule("r.jina.ai", "/https://www.alphaxiv.org/abs/2401.12345")
            == "host:alphaxiv.org"
        )

    @pytest.mark.parametrize(
        ("host", "target"),
        [
            ("r.jina.ai", "/https://arxiv.org/abs/2401.12346"),
            ("r.jina.ai", "/https://arxiv.org/abs/1706.03762"),
            ("r.jina.ai", "/https://notarxiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv.org.mirror.test/abs/2401.12345"),
            ("files.test", "/numpy-1.26.4.tar.gz?v=2401.12345"),
            ("search.test", "/search?q=2401.12345"),
            ("arxiv-org.translate.goog", "/abs/1706.03762"),
            ("translate.goog", "/abs/2401.12345"),
        ],
    )
    def test_other_addresses_inside_urls_stay_reachable(self, host, target):
        assert Policy([PAPER], ["alphaxiv.org"]).url_rule(host, target) is None


class TestMirrors:
    """Services that serve a listed repository by name are refused like the repository."""

    REPO = "github.com/benchflow-ai/FrontierPhysics"
    DATASET = "huggingface.co/datasets/benchflow/frontierphysics-iclr"
    MODEL = "huggingface.co/benchflow/fp-reviewer"

    @pytest.mark.parametrize(
        ("host", "target"),
        [
            ("cdn.jsdelivr.net", "/gh/benchflow-ai/FrontierPhysics@main/README.md"),
            ("fastly.jsdelivr.net", "/gh/benchflow-ai/frontierphysics/tasks/x/task.md"),
            ("cdn.statically.io", "/gh/benchflow-ai/FrontierPhysics/main/README.md"),
            ("raw.githack.com", "/benchflow-ai/FrontierPhysics/main/README.md"),
            (
                "raw.githubusercontent.com",
                "/benchflow-ai/FrontierPhysics/main/README.md",
            ),
            (
                "codeload.github.com",
                "/benchflow-ai/FrontierPhysics/zip/refs/heads/main",
            ),
            ("api.github.com", "/repos/benchflow-ai/FrontierPhysics/contents/tasks"),
            (
                "hf-mirror.com",
                "/datasets/benchflow/frontierphysics-iclr/resolve/main/x.json",
            ),
            ("huggingface.co", "/api/datasets/benchflow/frontierphysics-iclr/parquet"),
            (
                "datasets-server.huggingface.co",
                "/rows?dataset=benchflow%2Ffrontierphysics-iclr&split=train",
            ),
            ("huggingface.co", "/api/models/benchflow/fp-reviewer"),
            ("hf-mirror.com", "/benchflow/fp-reviewer/resolve/main/config.json"),
        ],
    )
    def test_mirrors_of_a_listed_repository_are_refused(self, host, target):
        policy = Policy([self.REPO, self.DATASET, self.MODEL], [])
        assert policy.url_rule(host, target) is not None

    @pytest.mark.parametrize(
        ("listed", "host", "target"),
        [
            # A listed page inside a repository stays a listing of that page.
            (
                "github.com/numpy/numpy/issues/1",
                "cdn.jsdelivr.net",
                "/gh/numpy/numpy@main/README.md",
            ),
            (REPO, "cdn.jsdelivr.net", "/gh/numpy/numpy@main/README.md"),
            (
                DATASET,
                "datasets-server.huggingface.co",
                "/rows?dataset=stanfordnlp%2Fimdb",
            ),
        ],
    )
    def test_other_repositories_stay_reachable(self, listed, host, target):
        assert Policy([listed], []).url_rule(host, target) is None


class TestNamedPages:
    @pytest.mark.parametrize(
        "text",
        [
            "url=https%3A%2F%2Farxiv.org%2Fabs%2F2401.12345&format=markdown",
            '{"url": "https://arxiv.org/abs/2401.12345"}',
            "fetch arxiv\uff0eorg/abs/2401.12345 now",
            "https://ar\u00adxiv.org/abs/2401.12345",
        ],
    )
    def test_a_listed_page_named_in_text_is_found(self, text):
        assert (
            Policy([PAPER], []).named_page_rule(text) == "url:arxiv.org/abs/2401.12345"
        )

    def test_other_text_passes(self):
        policy = Policy([PAPER], [])
        assert (
            policy.named_page_rule("see arxiv.org/abs/1706.03762 and 2401.12345")
            is None
        )
