//! Services on Docker: one container per service on a private bridge
//! network.
//!
//! The network is `--internal` (no route out) and the bridge gets no address
//! on the host (`com.docker.network.bridge.inhibit_ipv4`), so a container on
//! it cannot reach the host's ports through the gateway, which a plain
//! internal network allows. A service is reachable there by its name
//! (`--network-alias`). The agent's sandbox keeps its own network (the
//! default bridge when it is open) and joins this one too; with no network
//! it is on this one alone.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use super::cli::{Cli, EnvArgs};
use super::run::container_name;
use super::{DockerProvider, DockerSandbox};
use crate::owner::{self, Verdict};
use crate::services::{ServiceHost, ServiceSpec};
use crate::{NetworkPolicy, OWNER_LABEL, Sandbox, SandboxError, SandboxSpec};

/// How long a service may take to reach `running`.
const START_TIMEOUT: Duration = Duration::from_secs(60);
const CLI_TIMEOUT: Option<Duration> = Some(Duration::from_secs(120));

fn label_args(labels: &BTreeMap<String, String>) -> Vec<String> {
    labels
        .iter()
        .flat_map(|(k, v)| ["--label".to_string(), format!("{k}={v}")])
        .collect()
}

fn failed(cli: &Cli, what: &str, out: &super::cli::Output) -> SandboxError {
    cli.daemon_problem(out).unwrap_or_else(|| {
        SandboxError::Start(format!(
            "{what} failed: {}. This is a problem with the {} host; check `{} info`.",
            out.message(),
            cli.name,
            cli.name
        ))
    })
}

#[async_trait]
impl ServiceHost for DockerProvider {
    async fn create_network(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
    ) -> Result<String, SandboxError> {
        let net = container_name(&format!("net-{name}"));
        let mut all = owner::labels();
        all.extend(labels.clone());
        let mut args = vec![
            "network".to_string(),
            "create".to_string(),
            "--internal".to_string(),
            "--opt".to_string(),
            "com.docker.network.bridge.inhibit_ipv4=true".to_string(),
        ];
        args.extend(label_args(&all));
        args.push(net.clone());
        let out = self.cli.run(&args, CLI_TIMEOUT).await?;
        if !out.ok() {
            return Err(failed(&self.cli, "creating the services' network", &out));
        }
        Ok(net)
    }

    async fn start_service(
        &self,
        network: &str,
        spec: &ServiceSpec,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let name = container_name(&format!("svc-{}", spec.name));
        let env = EnvArgs::new(&spec.env)?;
        let mut labels = owner::labels();
        labels.insert("ai.benchflow.service".into(), spec.name.clone());
        labels.extend(spec.labels.clone());
        let mut args = vec![
            "run".to_string(),
            "-d".to_string(),
            "--name".to_string(),
            name.clone(),
            "--network".to_string(),
            network.to_string(),
            "--network-alias".to_string(),
            spec.name.clone(),
        ];
        args.extend(label_args(&labels));
        if let Some(p) = spec.pids_limit.filter(|p| *p > 0) {
            args.push("--pids-limit".into());
            args.push(p.to_string());
        }
        args.extend(env.args.iter().cloned());
        args.push(spec.image.reference.clone());
        args.extend(spec.command.iter().cloned());
        let mut cmd = self.cli.command(&args);
        for (k, v) in &env.process_env {
            cmd.env(k, v);
        }
        let out = self
            .cli
            .run_command(cmd, &args, Some(Duration::from_secs(300)))
            .await;
        env.close();
        let out = out?;
        if !out.ok() {
            remove(&self.cli, &name).await;
            return Err(failed(
                &self.cli,
                &format!("`{} run` for service {}", self.cli.name, spec.name),
                &out,
            ));
        }
        if let Err(e) = wait_running(&self.cli, &name, &spec.name).await {
            remove(&self.cli, &name).await;
            return Err(e);
        }
        // A service is the task's: its disk is not the agent's to watch.
        Ok(Box::new(DockerSandbox::new(
            self.cli.clone(),
            name,
            crate::DiskLimit::None,
        )))
    }

    async fn create_on_network(
        &self,
        spec: &SandboxSpec,
        network: &str,
    ) -> Result<Box<dyn Sandbox>, SandboxError> {
        let (name, limit) = super::run::start(&self.cli, spec).await?;
        let sandbox = DockerSandbox::new(self.cli.clone(), name.clone(), limit).with_spec(spec);
        if spec.network == NetworkPolicy::None {
            // A container on `none` joins no other network until it leaves it.
            let args = [
                "network".to_string(),
                "disconnect".to_string(),
                "none".to_string(),
                name.clone(),
            ];
            let out = self.cli.run(&args, CLI_TIMEOUT).await?;
            if !out.ok() {
                let _ = sandbox.destroy().await;
                return Err(failed(&self.cli, "leaving the `none` network", &out));
            }
        }
        let args = [
            "network".to_string(),
            "connect".to_string(),
            "--alias".to_string(),
            "main".to_string(),
            network.to_string(),
            name.clone(),
        ];
        let out = self.cli.run(&args, CLI_TIMEOUT).await?;
        if !out.ok() {
            let _ = sandbox.destroy().await;
            return Err(failed(&self.cli, "joining the services' network", &out));
        }
        Ok(Box::new(sandbox))
    }

    async fn logs(&self, id: &str) -> String {
        logs(&self.cli, id).await
    }

    async fn remove_network(&self, network: &str) -> Result<(), SandboxError> {
        remove_network(&self.cli, network).await
    }
}

async fn logs(cli: &Cli, id: &str) -> String {
    let args = [
        "logs".to_string(),
        "--tail".to_string(),
        "40".to_string(),
        id.to_string(),
    ];
    cli.run(&args, Some(Duration::from_secs(30)))
        .await
        .map(|o| format!("{}{}", o.stdout, o.stderr).trim().to_string())
        .unwrap_or_default()
}

async fn remove_network(cli: &Cli, network: &str) -> Result<(), SandboxError> {
    let args = ["network".to_string(), "rm".to_string(), network.to_string()];
    let out = cli.run(&args, CLI_TIMEOUT).await?;
    if out.ok() || out.message().to_lowercase().contains("not found") {
        return Ok(());
    }
    Err(SandboxError::Transport(format!(
        "could not remove network {network}: {}. Remove it with `{} network rm {network}`; \
         `bench` also sweeps leftovers at the start of the next run.",
        out.message(),
        cli.name
    )))
}

/// Poll a service until it runs; fail with its output if it exits.
async fn wait_running(cli: &Cli, id: &str, service: &str) -> Result<(), SandboxError> {
    let started = Instant::now();
    let args = [
        "inspect".to_string(),
        "--format".to_string(),
        "{{.State.Status}}\t{{.State.ExitCode}}".to_string(),
        id.to_string(),
    ];
    loop {
        let out = cli.run(&args, Some(Duration::from_secs(30))).await?;
        let line = out.stdout.trim().to_string();
        let (status, code) = line.split_once('\t').unwrap_or((line.as_str(), ""));
        match status {
            "running" => return Ok(()),
            "exited" | "dead" => {
                let logs = logs(cli, id).await;
                return Err(SandboxError::Start(format!(
                    "service {service} stopped right after starting (exit code {code}). Its \
                     output: {}. Check the service's image and command in the task.",
                    super::cli::tail_chars(&logs, 1000)
                )));
            }
            _ if started.elapsed() > START_TIMEOUT => {
                return Err(SandboxError::Start(format!(
                    "service {service} was still `{status}` after {}s",
                    START_TIMEOUT.as_secs()
                )));
            }
            _ => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}

async fn remove(cli: &Cli, id: &str) {
    let args = ["rm".to_string(), "-f".to_string(), id.to_string()];
    let _ = cli.run(&args, Some(Duration::from_secs(60))).await;
}

/// Remove networks that crashed runs left behind, by the rule containers
/// are swept by (see [`crate::owner`]). Returns how many were removed.
pub(super) async fn sweep_networks(cli: &Cli) -> usize {
    let list = [
        "network".to_string(),
        "ls".to_string(),
        "-q".to_string(),
        "--filter".to_string(),
        format!("label={OWNER_LABEL}={}", owner::scope()),
    ];
    let Ok(out) = cli.run(&list, Some(Duration::from_secs(60))).await else {
        return 0;
    };
    let mut removed = 0;
    for id in out.stdout.split_whitespace() {
        let inspect = [
            "network".to_string(),
            "inspect".to_string(),
            "--format".to_string(),
            "{{json .Labels}}".to_string(),
            id.to_string(),
        ];
        let Ok(o) = cli.run(&inspect, Some(Duration::from_secs(30))).await else {
            continue;
        };
        let labels: BTreeMap<String, String> =
            serde_json::from_str(o.stdout.trim()).unwrap_or_default();
        if owner::judge(|k| labels.get(k).map(String::as_str)).await == Verdict::Leftover
            && remove_network(cli, id).await.is_ok()
        {
            removed += 1;
        }
    }
    removed
}
