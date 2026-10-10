//! DSPy-RLM (`-a dspy-rlm`): DSPy's recursive language model, after
//! Harbor's `dspy-rlm` agent, run inside the sandbox as an SDK runner.
//!
//! Harbor runs DSPy on the host and bridges each of its tools to the
//! sandbox. BenchFlow runs the same loop inside the sandbox instead (DSPy
//! and Deno pinned in a venv, `runners/run_dspy_rlm.py`), so its tools act
//! on the task's files directly and the host needs neither Python nor Deno.
//! The model's Python still runs in DSPy's Deno/Pyodide REPL.
//!
//! The model is reached through BenchFlow's provider forwarder, as for any
//! headless agent on a provider: the headless harness starts the forwarder
//! as root in the sandbox, holding the provider's key, then runs the
//! manifest's command with the forwarder's address and a placeholder key.
//! [`DspyRlmAgent`] only insists on a model that names a provider. On a
//! gateway, the forwarder pins every request to the allowed upstreams
//! (never Baseten). The manifest (`runners/dspy-rlm.toml`) is not in the
//! agent registry, so the runner never runs without the forwarder.

use std::sync::OnceLock;

use async_trait::async_trait;
use benchflow_agent::AgentManifest;
use benchflow_agent::credentials::HostEnv;
use benchflow_agent::manifest::ManifestSource;
use benchflow_agent::providers::{self, Access, AccessError, ProviderRegistry, Route};
use sha2::{Digest, Sha256};

use crate::agent::{AgentContext, AgentOutput, ModelAgent};
use crate::forwarding::{ProviderAccess, login_problem, pin_problem, route_problem};
use crate::headless_agent::HeadlessAgent;
use crate::outcome::{Owner, Phase, Problem};

/// The agent's name.
pub const NAME: &str = "dspy-rlm";

/// Its manifest, embedded.
const MANIFEST: &str = include_str!("../../benchflow-agent/runners/dspy-rlm.toml");

/// Fills Deno's cache with Pyodide at install time (as root, with network),
/// so a trial's REPL starts without downloading it.
pub const PREWARM_LAYER: &str = "set -e; d=/opt/benchflow/agents/dspy-rlm; mkdir -p \"$d/deno\"; \
DENO_DIR=\"$d/deno\" \"$d/venv/bin/python\" -c \"from dspy.primitives.python_interpreter import PythonInterpreter as P; i = P(); r = i.execute('print(6 * 7)'); i.shutdown(); assert '42' in str(r), r; print('dspy-rlm: Pyodide cached')\"; \
chmod -R a+rX \"$d/deno\"";

/// The embedded manifest.
pub fn manifest() -> &'static AgentManifest {
    static PARSED: OnceLock<AgentManifest> = OnceLock::new();
    PARSED.get_or_init(|| {
        AgentManifest::parse(MANIFEST, ManifestSource::Builtin("dspy-rlm.toml"))
            .expect("the embedded dspy-rlm manifest is valid")
    })
}

/// Whether `name` means DSPy-RLM.
pub fn is_dspy_rlm(name: &str) -> bool {
    let m = manifest();
    name == m.name || m.aliases.iter().any(|a| a == name)
}

/// DSPy-RLM on a provider's model, through the forwarder.
pub struct DspyRlmAgent {
    headless: HeadlessAgent,
    access: Result<ProviderAccess, Problem>,
}

impl DspyRlmAgent {
    pub fn new(model: Option<&str>, effort: Option<&str>, host: &HostEnv) -> Self {
        Self::with_providers(model, effort, host, &ProviderRegistry::builtin())
    }

    /// [`Self::new`] with these providers.
    pub fn with_providers(
        model: Option<&str>,
        effort: Option<&str>,
        host: &HostEnv,
        registry: &ProviderRegistry,
    ) -> Self {
        let manifest = manifest().clone();
        let access = match model {
            None => Err(Problem::new(
                "R215",
                Owner::You,
                Phase::Setup,
                "dspy-rlm needs a model and none was given",
                "pass --model <provider>/<model>, such as --model vercel/deepseek/deepseek-v4-flash-0731",
            )),
            Some(m) => match providers::resolve(&manifest, Some(m), host, registry) {
                Ok(Access {
                    route: Route::Provider(mut route),
                    key: Some(key),
                    ..
                }) => match providers::pin_route(&mut route, host) {
                    Ok(pin) => Ok(ProviderAccess { route, key, pin }),
                    Err(e) => Err(pin_problem(&e)),
                },
                Ok(_) => Err(Problem::new(
                    "R215",
                    Owner::You,
                    Phase::Setup,
                    format!("dspy-rlm reaches its model through a provider, and `{m}` names none"),
                    "name the model as <provider>/<model>, such as vercel/deepseek/deepseek-v4-flash-0731 (`bench agents dspy-rlm` lists the providers)",
                )),
                Err(AccessError::Login(e)) => Err(login_problem(&e)),
                Err(AccessError::Route(e)) => Err(route_problem(&e)),
            },
        };
        Self {
            headless: HeadlessAgent::with_providers(manifest, model, effort, host, registry),
            access,
        }
    }

    /// Where the key comes from, as people see it (never a value).
    pub fn login_label(&self) -> Option<String> {
        self.access
            .as_ref()
            .ok()
            .map(|a| a.key.login(&a.route.describe()).label)
    }
}

#[async_trait]
impl ModelAgent for DspyRlmAgent {
    fn name(&self) -> &str {
        NAME
    }

    fn model(&self) -> Option<&str> {
        self.headless.model()
    }

    /// The venv (DSPy, Deno), Pyodide in Deno's cache, and Node.js for the
    /// forwarder.
    fn install_layers(&self) -> Vec<String> {
        // The headless harness adds Node.js for the forwarder.
        let mut layers = self.headless.install_layers();
        layers.insert(1.min(layers.len()), PREWARM_LAYER.to_string());
        layers
    }

    fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        for layer in self.install_layers() {
            h.update(layer.as_bytes());
        }
        format!("{}:{}", NAME, &hex::encode(h.finalize())[..12])
    }

    fn preflight(&self) -> Result<(), Problem> {
        self.access.as_ref().map(|_| ()).map_err(Clone::clone)?;
        self.headless.preflight()
    }

    async fn run(&self, ctx: AgentContext<'_>) -> AgentOutput {
        if let Err(p) = &self.access {
            return AgentOutput {
                problem: Some(p.clone()),
                ..AgentOutput::default()
            };
        }
        // The headless harness starts the forwarder (root, holding the key),
        // pins a gateway's upstreams, and reads a rejected key from its log.
        let mut out = self.headless.run(ctx).await;
        if let Some(label) = self.login_label() {
            out.login = Some(label);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn host(vars: &[(&str, &str)]) -> HostEnv {
        HostEnv::new(
            vars.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            None,
        )
    }

    #[test]
    fn the_embedded_manifest_is_a_headless_runner_with_no_login() {
        let m = manifest();
        assert_eq!(m.name, "dspy-rlm");
        assert!(is_dspy_rlm("dspy-rlm") && is_dspy_rlm("dspy"));
        assert!(m.install_script().contains("dspy==3.4.0"));
        assert!(m.install_script().contains("deno==2.9.7"));
        assert!(
            m.headless
                .as_ref()
                .unwrap()
                .command
                .contains("run_dspy_rlm.py")
        );
    }

    #[test]
    fn a_provider_model_runs_through_the_forwarder_and_others_are_refused() {
        // OpenRouter: a provider with no upstream list to read.
        let h = host(&[("OPENROUTER_API_KEY", "sk-or-1234567890")]);
        let a = DspyRlmAgent::new(Some("openrouter/deepseek/deepseek-v4-flash"), None, &h);
        assert!(a.preflight().is_ok(), "{:?}", a.preflight());
        let layers = a.install_layers();
        assert_eq!(layers.len(), 3);
        assert!(layers[1].contains("Pyodide cached"));
        assert!(layers[2].contains("/opt/benchflow/node"));
        assert!(a.needs_network());
        assert!(a.login_label().unwrap().contains("OPENROUTER_API_KEY"));

        let p = DspyRlmAgent::new(Some("deepseek-v4-flash"), None, &h)
            .preflight()
            .unwrap_err();
        assert_eq!((p.code.as_str(), p.owner), ("R215", Owner::You));
        assert!(p.next.contains("<provider>/<model>"));
        let p = DspyRlmAgent::new(None, None, &h).preflight().unwrap_err();
        assert_eq!(p.code, "R215");
        let p = DspyRlmAgent::new(Some("openai/gpt-5-mini"), None, &host(&[]))
            .preflight()
            .unwrap_err();
        assert_eq!(p.code, "R211");
    }

    #[test]
    fn the_access_check_and_the_run_use_the_same_providers() {
        use benchflow_agent::manifest::ManifestSource;
        use benchflow_agent::providers::Provider;
        let mut registry = ProviderRegistry::builtin();
        registry.add(
            Provider::parse(
                "name = \"testprov\"\n[openai]\nbase_url = \"https://llm.example/v1\"\napis = [\"chat\"]\n[key]\nenv = [\"TESTPROV_KEY\"]\n",
                ManifestSource::File("providers/testprov.toml".into()),
            )
            .unwrap(),
        );
        let h = host(&[("TESTPROV_KEY", "sk-testprov-1234567890")]);
        let a = DspyRlmAgent::with_providers(Some("testprov/some-model"), None, &h, &registry);
        assert!(a.preflight().is_ok(), "{:?}", a.preflight());
        // The run's forwarder holds the key the check found.
        let label = a.headless.login_label().unwrap();
        assert!(label.contains("held by BenchFlow's forwarder"), "{label}");
    }
}
