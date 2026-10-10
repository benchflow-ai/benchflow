//! Uploading a build context to Daytona's object storage.
//!
//! Daytona builds a Dockerfile from context archives it finds in an S3
//! bucket at `<organization>/<hash>/context.tar`. `GET
//! /object-storage/push-access` hands out short-lived credentials for that
//! bucket; this module signs `HEAD` and `PUT` requests with them (AWS
//! Signature Version 4, unsigned payload over HTTPS).

use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::StatusCode;
use sha2::{Digest, Sha256};

use super::api::Secret;
use super::types::PushAccess;
use crate::SandboxError;
use crate::redact::Redactor;

/// A bucket and the credentials to write to it.
pub(crate) struct Bucket {
    http: reqwest::Client,
    access_key: String,
    secret: Secret,
    session_token: Secret,
    region: String,
    /// `https://<host>`, without a trailing `/`.
    endpoint: String,
    bucket: String,
    /// `<bucket>.<host>` addressing (AWS) instead of `<host>/<bucket>`.
    virtual_host: bool,
    redactor: Redactor,
}

impl Bucket {
    pub(crate) fn new(access: PushAccess, http: reqwest::Client) -> Self {
        let endpoint = access.storage_url.trim_end_matches('/').to_string();
        let host = host_of(&endpoint);
        let dns_safe = !access.bucket.contains('.')
            && access
                .bucket
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        let virtual_host = host.ends_with(".amazonaws.com") && dns_safe;
        let mut redactor = Redactor::new();
        redactor.add(access.secret.expose().to_string());
        redactor.add(access.session_token.expose().to_string());
        Self {
            http,
            access_key: access.access_key,
            secret: access.secret,
            session_token: access.session_token,
            region: access.region.unwrap_or_else(|| "us-east-1".to_string()),
            endpoint,
            bucket: access.bucket,
            virtual_host,
            redactor,
        }
    }

    /// The URL and the canonical path of `key`.
    fn locate(&self, key: &str) -> (String, String, String) {
        let host = host_of(&self.endpoint);
        let scheme = if self.endpoint.starts_with("http://") {
            "http"
        } else {
            "https"
        };
        let encoded_key = uri_encode(key, false);
        if self.virtual_host {
            let host = format!("{}.{host}", self.bucket);
            let path = format!("/{encoded_key}");
            (format!("{scheme}://{host}{path}"), host, path)
        } else {
            let path = format!("/{}/{encoded_key}", uri_encode(&self.bucket, true));
            (format!("{scheme}://{host}{path}"), host.to_string(), path)
        }
    }

    fn signed(
        &self,
        method: reqwest::Method,
        key: &str,
        now: DateTime<Utc>,
    ) -> reqwest::RequestBuilder {
        let (url, host, path) = self.locate(key);
        let headers = sign(
            &Credentials {
                access_key: &self.access_key,
                secret: self.secret.expose(),
                session_token: Some(self.session_token.expose()),
                region: &self.region,
                service: "s3",
            },
            &Request {
                method: method.as_str(),
                host: &host,
                path: &path,
                payload_hash: "UNSIGNED-PAYLOAD",
                extra: &[],
            },
            now,
        );
        let mut request = self.http.request(method, url);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        request
    }

    /// Whether `key` already exists. Anything but a clear yes counts as no,
    /// so the caller uploads.
    pub(crate) async fn exists(&self, key: &str) -> bool {
        let request = self
            .signed(reqwest::Method::HEAD, key, Utc::now())
            .timeout(Duration::from_secs(30));
        matches!(request.send().await, Ok(r) if r.status() == StatusCode::OK)
    }

    /// Upload `file` (`len` bytes) as `key`.
    pub(crate) async fn put(
        &self,
        key: &str,
        file: &std::fs::File,
        len: u64,
    ) -> Result<(), SandboxError> {
        let mut last = String::new();
        for attempt in 1..=3u32 {
            let mut handle = file
                .try_clone()
                .map_err(|e| SandboxError::io("reopening the build context archive", e))?;
            std::io::Seek::seek(&mut handle, std::io::SeekFrom::Start(0))
                .map_err(|e| SandboxError::io("rewinding the build context archive", e))?;
            let body = reqwest::Body::from(tokio::fs::File::from_std(handle));
            let request = self
                .signed(reqwest::Method::PUT, key, Utc::now())
                .header(reqwest::header::CONTENT_LENGTH, len)
                .header(reqwest::header::CONTENT_TYPE, "application/x-tar")
                .timeout(Duration::from_secs(600))
                .body(body);
            match request.send().await {
                Ok(r) if r.status().is_success() => return Ok(()),
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    last = format!("HTTP {status}: {}", s3_error(&body));
                    if !(status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS) {
                        break;
                    }
                }
                Err(e) => last = e.to_string(),
            }
            tokio::time::sleep(Duration::from_millis(500 * u64::from(attempt))).await;
        }
        Err(SandboxError::Image {
            name: String::new(),
            message: format!(
                "uploading the build context to Daytona's storage failed ({}). This is a problem \
                 between BenchFlow and Daytona, not the task; run again, and check for a newer \
                 BenchFlow if it persists.",
                self.redactor.redact(&last)
            ),
            log_tail: String::new(),
            task_fault: false,
        })
    }
}

/// The `<Code>` and `<Message>` of an S3 XML error.
fn s3_error(body: &str) -> String {
    let field = |tag: &str| {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = body.find(&open)? + open.len();
        let end = body[start..].find(&close)? + start;
        Some(body[start..end].to_string())
    };
    match (field("Code"), field("Message")) {
        (Some(code), Some(message)) => format!("{code}: {message}"),
        (Some(code), None) => code,
        _ => body.chars().take(300).collect(),
    }
}

fn host_of(endpoint: &str) -> &str {
    let rest = endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))
        .unwrap_or(endpoint);
    rest.split('/').next().unwrap_or(rest)
}

/// Signing inputs: who signs, and for which region and service.
pub(crate) struct Credentials<'a> {
    pub access_key: &'a str,
    pub secret: &'a str,
    pub session_token: Option<&'a str>,
    pub region: &'a str,
    pub service: &'a str,
}

/// The request being signed. `path` is already URI-encoded.
pub(crate) struct Request<'a> {
    pub method: &'a str,
    pub host: &'a str,
    pub path: &'a str,
    pub payload_hash: &'a str,
    /// Other headers to sign, as (lowercase name, value).
    pub extra: &'a [(&'a str, &'a str)],
}

/// The headers that sign `request` at `now`: `host`, `x-amz-date`,
/// `x-amz-content-sha256`, the session token if any, and `authorization`.
pub(crate) fn sign(
    creds: &Credentials<'_>,
    request: &Request<'_>,
    now: DateTime<Utc>,
) -> Vec<(String, String)> {
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let date = now.format("%Y%m%d").to_string();
    let mut headers: Vec<(String, String)> = vec![
        ("host".into(), request.host.to_string()),
        (
            "x-amz-content-sha256".into(),
            request.payload_hash.to_string(),
        ),
        ("x-amz-date".into(), amz_date.clone()),
    ];
    if let Some(token) = creds.session_token {
        headers.push(("x-amz-security-token".into(), token.to_string()));
    }
    for (name, value) in request.extra {
        headers.push((name.to_string(), value.to_string()));
    }
    headers.sort_by(|a, b| a.0.cmp(&b.0));
    let canonical_headers: String = headers
        .iter()
        .map(|(k, v)| format!("{k}:{}\n", v.trim()))
        .collect();
    let signed_headers = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_request = format!(
        "{}\n{}\n\n{canonical_headers}\n{signed_headers}\n{}",
        request.method, request.path, request.payload_hash
    );
    let scope = format!("{date}/{}/{}/aws4_request", creds.region, creds.service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );
    let key = signing_key(creds.secret, &date, creds.region, creds.service);
    let signature = hex::encode(hmac_sha256(&key, string_to_sign.as_bytes()));
    headers.push((
        "authorization".into(),
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
            creds.access_key
        ),
    ));
    // `host` is set by the HTTP client from the URL.
    headers.retain(|(k, _)| k != "host");
    headers
}

/// The SigV4 signing key for a day, region and service.
pub(crate) fn signing_key(secret: &str, date: &str, region: &str, service: &str) -> [u8; 32] {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

/// HMAC-SHA256 (RFC 2104).
pub(crate) fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut block = [0u8; BLOCK];
    if key.len() > BLOCK {
        block[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        block[..key.len()].copy_from_slice(key);
    }
    let mut inner = Sha256::new();
    inner.update(block.map(|b| b ^ 0x36));
    inner.update(data);
    let inner = inner.finalize();
    let mut outer = Sha256::new();
    outer.update(block.map(|b| b ^ 0x5c));
    outer.update(inner);
    outer.finalize().into()
}

/// URI-encode per SigV4: unreserved characters stay, everything else is
/// `%XX`; `/` stays unless `encode_slash`.
pub(crate) fn uri_encode(text: &str, encode_slash: bool) -> String {
    let mut out = String::new();
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) || (b == b'/' && !encode_slash) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn hmac_matches_rfc_4231() {
        // Test case 2.
        assert_eq!(
            hex::encode(hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // Test case 6: a key longer than the block.
        assert_eq!(
            hex::encode(hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn signing_key_matches_the_aws_documentation() {
        // "Deriving a signing key" example from the AWS SigV4 documentation.
        let key = signing_key(
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            "20120215",
            "us-east-1",
            "iam",
        );
        assert_eq!(
            hex::encode(key),
            "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d"
        );
    }

    #[test]
    fn signature_matches_the_s3_documentation_example() {
        // "GET Object" example from the S3 SigV4 header-signing documentation.
        let headers = sign(
            &Credentials {
                access_key: "AKIAIOSFODNN7EXAMPLE",
                secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                session_token: None,
                region: "us-east-1",
                service: "s3",
            },
            &Request {
                method: "GET",
                host: "examplebucket.s3.amazonaws.com",
                path: "/test.txt",
                payload_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                extra: &[("range", "bytes=0-9")],
            },
            Utc.with_ymd_and_hms(2013, 5, 24, 0, 0, 0).unwrap(),
        );
        let auth = &headers
            .iter()
            .find(|(k, _)| k == "authorization")
            .unwrap()
            .1;
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, \
             SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, \
             Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
        );
        assert!(!headers.iter().any(|(k, _)| k == "host"));
    }

    #[test]
    fn uri_encoding_follows_sigv4() {
        assert_eq!(
            uri_encode("org/abc-1.2_3~/context.tar", false),
            "org/abc-1.2_3~/context.tar"
        );
        assert_eq!(uri_encode("a b+c", false), "a%20b%2Bc");
        assert_eq!(uri_encode("a/b", true), "a%2Fb");
    }

    #[test]
    fn s3_errors_are_short() {
        assert_eq!(
            s3_error("<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"),
            "AccessDenied: Access Denied"
        );
    }

    #[test]
    fn aws_buckets_use_virtual_hosts() {
        let access = |url: &str, bucket: &str| PushAccess {
            access_key: "AK".into(),
            secret: Secret::new("S".repeat(10)),
            session_token: Secret::new("T".repeat(10)),
            storage_url: url.into(),
            organization_id: "org".into(),
            bucket: bucket.into(),
            region: None,
        };
        let aws = Bucket::new(
            access("https://s3.us-east-1.amazonaws.com", "prod-ctx"),
            reqwest::Client::new(),
        );
        assert_eq!(
            aws.locate("org/h/context.tar").0,
            "https://prod-ctx.s3.us-east-1.amazonaws.com/org/h/context.tar"
        );
        let minio = Bucket::new(access("http://minio:9000/", "ctx"), reqwest::Client::new());
        assert_eq!(
            minio.locate("org/h/context.tar").0,
            "http://minio:9000/ctx/org/h/context.tar"
        );
    }
}
