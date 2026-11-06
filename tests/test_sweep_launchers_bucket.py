"""Guards FF-sw-launchers against the cleanup regression at 363d815b."""

from unittest.mock import MagicMock

from benchflow.sandbox.agentcore_builder import CodeBuildBuilder


def test_expiration_requires_owned_object_tag():
    s3 = MagicMock()
    s3.get_bucket_lifecycle_configuration.return_value = {"Rules": []}
    builder = CodeBuildBuilder(lambda _: s3, "1", "us-east-1")
    builder._ensure_bucket()
    rules = s3.put_bucket_lifecycle_configuration.call_args.kwargs[
        "LifecycleConfiguration"
    ]["Rules"]
    rule = next(r for r in rules if r["ID"] == "benchflow-expire-build-contexts")
    assert rule["Filter"] == {
        "And": {
            "Prefix": "contexts/",
            "Tags": [{"Key": "benchflow-build-context", "Value": "1"}],
        }
    }


async def test_uploaded_context_carries_expiration_receipt(tmp_path, monkeypatch):
    """Guards FF-sw-launchers: the lifecycle predicate and upload agree."""
    from benchflow.sandbox.agentcore_builder import BuildRequest

    s3 = MagicMock()
    builder = CodeBuildBuilder(lambda _: s3, "1", "us-east-1")
    monkeypatch.setattr(builder, "_package", lambda request: b"dummy")
    monkeypatch.setattr(builder, "_ensure_bucket", lambda: None)
    monkeypatch.setattr(builder, "_ensure_project", lambda: None)
    monkeypatch.setattr(builder, "_role_arn", lambda: "dummy")
    # Stop after upload; cleanup still executes and no provider is contacted.
    s3.start_build.side_effect = RuntimeError("dummy stop")
    request = BuildRequest(
        context_dir=tmp_path,
        dockerfile_text="FROM scratch",
        shim_text="pass",
        image_uri="repo:test",
        registry="repo",
        region="us-east-1",
        force_build=False,
        timeout_sec=1,
    )
    import pytest

    with pytest.raises(RuntimeError, match="dummy stop"):
        await builder.build_and_push(request)
    assert s3.put_object.call_args.kwargs["Tagging"] == "benchflow-build-context=1"
