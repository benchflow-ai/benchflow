//! Checking a provider key with the provider's free request.

use std::time::Duration;

use super::key::ProviderKey;
use super::{Auth, Provider};

/// What the provider said about a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// It accepted the key.
    Accepted,
    /// It refused the key (HTTP 401 or 403).
    Rejected { status: u16 },
    /// Something else happened; the key may still be good.
    Inconclusive { detail: String },
    /// The provider declares no free request to check with.
    NoCheck,
}

impl CheckOutcome {
    /// One line for people.
    pub fn describe(&self) -> String {
        match self {
            Self::Accepted => "the provider accepts the key".to_string(),
            Self::Rejected { status } => format!("the provider rejects the key (HTTP {status})"),
            Self::Inconclusive { detail } => format!("could not check the key: {detail}"),
            Self::NoCheck => {
                "not checked (the provider has no free request to check with)".to_string()
            }
        }
    }
}

/// Send `provider`'s check request with `key`: a `GET` that needs a valid
/// key and spends nothing. Redirects are not followed, so the key goes to the
/// provider's own address only, and the answer's body is never read.
pub async fn check_key(provider: &Provider, key: &ProviderKey, timeout: Duration) -> CheckOutcome {
    let Some(check) = &provider.check else {
        return CheckOutcome::NoCheck;
    };
    let Some(endpoint) = provider.endpoint(check.endpoint) else {
        return CheckOutcome::NoCheck;
    };
    let client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .build()
    {
        Ok(client) => client,
        Err(e) => {
            return CheckOutcome::Inconclusive {
                detail: format!("no HTTP client: {e}"),
            };
        }
    };
    let url = format!("{}{}", endpoint.base_url, check.path);
    let mut request = client.get(&url).header(
        "user-agent",
        concat!("benchflow/", env!("CARGO_PKG_VERSION")),
    );
    request = match endpoint.auth {
        Auth::Bearer => request.bearer_auth(key.value()),
        Auth::XApiKey => request.header("x-api-key", key.value()),
    };
    for (name, value) in &check.headers {
        request = request.header(name.as_str(), value.as_str());
    }
    match request.send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            match status {
                200..=299 => CheckOutcome::Accepted,
                401 | 403 => CheckOutcome::Rejected { status },
                _ => CheckOutcome::Inconclusive {
                    detail: format!("{url} answered HTTP {status}"),
                },
            }
        }
        Err(e) => CheckOutcome::Inconclusive {
            detail: if e.is_timeout() {
                format!("{url} did not answer within {}s", timeout.as_secs())
            } else {
                format!("{url} could not be reached")
            },
        },
    }
}
