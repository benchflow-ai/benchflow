//! Services: containers beside a sandbox, on a private network the sandbox
//! reaches them on by name.
//!
//! A backend that can run them returns a [`ServiceHost`] from
//! [`Provider::services`](crate::Provider::services). The network is private
//! in both directions: it has no route to the internet, and the host has no
//! address on it, so neither a service nor the sandbox can reach the host
//! through it. Each service is a [`Sandbox`] the caller can run commands in
//! (readiness checks, hooks) and must destroy, then remove the network.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Image, Sandbox, SandboxError, SandboxSpec};

/// One service to start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSpec {
    /// The host name the sandbox reaches it by.
    pub name: String,
    pub image: Image,
    /// argv; empty runs the image's own command.
    pub command: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Most processes the service may hold at once.
    pub pids_limit: Option<u32>,
    /// Extra labels (the trial's name).
    pub labels: BTreeMap<String, String>,
}

/// A backend that runs services beside a sandbox.
#[async_trait]
pub trait ServiceHost: Send + Sync {
    /// Create a private network named after `name`; returns its id.
    async fn create_network(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
    ) -> Result<String, SandboxError>;

    /// Start a service on `network`, reachable there as `spec.name`.
    async fn start_service(
        &self,
        network: &str,
        spec: &ServiceSpec,
    ) -> Result<Box<dyn Sandbox>, SandboxError>;

    /// Start a sandbox that also joins `network`. Its own network policy
    /// still holds: with [`NetworkPolicy::None`](crate::NetworkPolicy::None)
    /// the private network is its only one.
    async fn create_on_network(
        &self,
        spec: &SandboxSpec,
        network: &str,
    ) -> Result<Box<dyn Sandbox>, SandboxError>;

    /// The last lines a service printed, for error messages.
    async fn logs(&self, id: &str) -> String;

    /// Remove the network (after every container on it is gone).
    async fn remove_network(&self, network: &str) -> Result<(), SandboxError>;
}
