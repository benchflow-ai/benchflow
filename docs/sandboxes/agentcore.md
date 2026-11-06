# Amazon Bedrock AgentCore

```bash
uv tool install --python 3.12 --upgrade 'benchflow[sandbox-agentcore]'
export BENCHFLOW_AGENTCORE_ROLE_ARN="arn:aws:iam::<account>:role/<role>"
export BENCHFLOW_AGENTCORE_REGION="us-west-2"   # optional; this is the default
bench run tasks/ -a claude-agent-acp -s agentcore -j 32
```

AgentCore runs each trial in an AgentCore Runtime microVM on AWS. The model proxy runs inside it.

## Permissions

| Who | Needs |
|---|---|
| The execution role (`BENCHFLOW_AGENTCORE_ROLE_ARN`) | Assumable by `bedrock-agentcore.amazonaws.com`; pull from ECR; write CloudWatch logs |
| Your own credentials | `bedrock-agentcore:*` and ECR push |

## How images are built

AgentCore cannot run a container image directly. BenchFlow builds the task image for `linux/arm64`, pushes it to ECR and registers it as a runtime. The repository is `benchflow-agentcore`; set `BENCHFLOW_AGENTCORE_ECR_REPOSITORY` to change it. An unchanged task reuses the pushed image and runtime.

BenchFlow also adds a small HTTP responder to the image as its entrypoint. AgentCore runs commands only in a container that answers `GET /ping` on port 8080.

BenchFlow builds with a local Docker daemon when one is running, and otherwise on AWS CodeBuild. CodeBuild needs a service role:

```bash
export BENCHFLOW_AGENTCORE_CODEBUILD_ROLE_ARN="arn:aws:iam::<account>:role/<build-role>"
```

That role must be assumable by `codebuild.amazonaws.com`, push to ECR, read the build bucket and write CloudWatch logs. Your credentials need `codebuild:*` and S3 access.

On first use BenchFlow creates the CodeBuild project and a bucket named `benchflow-agentcore-build-<account>-<region>` (build contexts expire after a day). `BENCHFLOW_AGENTCORE_BUILD_BUCKET` overrides the bucket. `BENCHFLOW_AGENTCORE_BUILDER=docker|codebuild|auto` forces a builder. A build happens once per distinct task image, not once per trial. CodeBuild runs on Graviton, so `linux/arm64` is native: a remote build takes about 40 s, against about 20 s locally on Apple silicon.

## Run inside a VPC

Runtimes are `PUBLIC` by default. To place them in your VPC:

```bash
export BENCHFLOW_AGENTCORE_NETWORK_MODE="VPC"
export BENCHFLOW_AGENTCORE_SUBNETS="subnet-0a1b2c3d4e5f67890,subnet-01234567"
export BENCHFLOW_AGENTCORE_SECURITY_GROUPS="sg-0a1b2c3d4e5f67890"
```

- Both lists are required in `VPC` mode, up to 16 ids each.
- Setting them without `BENCHFLOW_AGENTCORE_NETWORK_MODE=VPC` is an error, not a silent fallback to `PUBLIC`.
- The subnets must reach ECR, CloudWatch Logs, and whatever the task and model proxy need.
- VPC and `PUBLIC` runtimes for the same image are never shared.

## Limits

| Limit | Detail |
|---|---|
| Architecture | `linux/arm64` only |
| Multi-container tasks | Not supported |
| `no-network`, `denylist`, `allowlist` | Refused: AgentCore's network mode is only `PUBLIC` or `VPC`, and BenchFlow cannot verify a VPC's routes |
| Separate verifier sandbox | Not supported |
| Snapshots and cached install layers | None |
| Image size | 2 GB, a fixed AWS limit; BenchFlow fails the build and names the cap |
| Session timeouts | Idle: the task's agent timeout, at least 15 minutes. Total: 8 hours, AgentCore's maximum. `BENCHFLOW_AGENTCORE_IDLE_TIMEOUT_SEC` and `BENCHFLOW_AGENTCORE_MAX_LIFETIME_SEC` override them (60–28,800 s each) |

Run heavy images (LaTeX, Playwright, large model snapshots) on Docker or Daytona.

## Large runs

A trial is a session, not a runtime. One runtime hosts many concurrent microVMs, so concurrency is bounded by sessions, not by the number of images.

| AWS quota | Default | Adjustable |
|---|---|---|
| Active session workloads per account | 5,000 | Yes |
| Runtimes per account | 100 | Yes |
| Maximum image size | 2 GB | No |

Each distinct task image needs its own runtime. A run with more than 100 distinct images needs a quota increase first. The first limit a large run usually meets is session creation, 400 per minute.

## Clean up

Runtimes are shared, so they outlive the run that created them. Remove old ones by age:

```bash
bench sandbox cleanup --max-age 1440
```

Only runtimes tagged `benchflow-managed` are deleted. Cleanup does nothing unless `BENCHFLOW_AGENTCORE_ROLE_ARN` is set.

## Next steps

- Compare sandboxes: [Sandboxes](./index.md).
- What the sandbox guarantees: [Sandbox security](./security.md).
- Run in the cloud with Compose and network policies: [Daytona](./daytona.md).
