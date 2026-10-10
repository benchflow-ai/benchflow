//! The desktop in the sandbox: a virtual display, a window manager, a
//! browser, and the actions computer-1 takes on them.
//!
//! Ported from Harbor's computer-1 runtime (Apache-2.0,
//! `harbor/agents/computer_1/runtime.py`): the same display (`:1`), key
//! aliases, blocked shortcuts and URL schemes, xdotool command lines for each
//! action, URL-bar navigation, and one-shot zoom crops. Everything runs as
//! shell commands through the sandbox API, as the agent user; nothing is
//! installed, so the task's image must have the desktop (see
//! [`REQUIRED`]).

use std::collections::BTreeMap;
use std::time::Duration;

use benchflow_sandbox::{Exec, ExecOutput, Sandbox, SandboxError};

use crate::shell::quote as q;

/// The X display the desktop runs on (Harbor's).
pub const DISPLAY: &str = ":1";
/// Where the desktop's logs and screenshots go in the sandbox.
pub const RUNTIME_DIR: &str = "/tmp/computer-1";

/// Programs the image must have: the display, the input tool, and a way to
/// take a screenshot.
pub const REQUIRED: &str = "Xvfb, xdotool, and ImageMagick's import (or scrot)";

/// Harbor's key aliases (lowercase name to xdotool keysym).
const KEY_ALIASES: &[(&str, &str)] = &[
    ("alt", "alt"),
    ("arrowdown", "Down"),
    ("arrowleft", "Left"),
    ("arrowright", "Right"),
    ("arrowup", "Up"),
    ("backspace", "BackSpace"),
    ("cmd", "super"),
    ("command", "super"),
    ("control", "ctrl"),
    ("ctrl", "ctrl"),
    ("delete", "Delete"),
    ("down", "Down"),
    ("end", "End"),
    ("enter", "Return"),
    ("esc", "Escape"),
    ("escape", "Escape"),
    ("home", "Home"),
    ("insert", "Insert"),
    ("left", "Left"),
    ("meta", "super"),
    ("option", "alt"),
    ("pagedown", "Next"),
    ("pageup", "Prior"),
    ("return", "Return"),
    ("right", "Right"),
    ("shift", "shift"),
    ("space", "space"),
    ("spacebar", "space"),
    ("tab", "Tab"),
    ("up", "Up"),
];

/// Shortcuts that open developer tools or page source (Harbor blocks them).
const BLOCKED_COMBOS: &[&str] = &[
    "ctrl+u",
    "ctrl+shift+i",
    "ctrl+shift+j",
    "ctrl+shift+c",
    "f12",
    "control+u",
    "control+shift+i",
    "control+shift+j",
    "control+shift+c",
];

const BLOCKED_SCHEMES: &[&str] = &["view-source:", "devtools://", "chrome-devtools://"];

/// One action the model asked for, as Harbor's `ComputerAction`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Action {
    pub kind: String,
    pub x: Option<i64>,
    pub y: Option<i64>,
    pub end_x: Option<i64>,
    pub end_y: Option<i64>,
    pub text: Option<String>,
    pub keys: Option<Vec<String>>,
    pub url: Option<String>,
    pub scroll_x: Option<i64>,
    pub scroll_y: Option<i64>,
    pub button: Option<String>,
    pub modifier: Option<String>,
    pub duration: Option<f64>,
    pub zoom_region: Option<[i64; 4]>,
    pub result: Option<String>,
}

/// The actions that end the run (twice in a row).
pub const TERMINAL: &[&str] = &["done", "answer", "terminate"];

impl Action {
    pub fn is_terminal(&self) -> bool {
        TERMINAL.contains(&self.kind.as_str())
    }
}

/// What an action did, for the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// `ok`, `done`, `blocked`, or `error`.
    pub status: &'static str,
    /// Text for the model (empty for a plain success).
    pub text: String,
}

impl Outcome {
    fn ok() -> Self {
        Self {
            status: "ok",
            text: String::new(),
        }
    }
    fn blocked(text: &str) -> Self {
        Self {
            status: "blocked",
            text: text.to_string(),
        }
    }
    fn error(text: impl Into<String>) -> Self {
        Self {
            status: "error",
            text: text.into(),
        }
    }
}

fn key(name: &str) -> String {
    let parts: Vec<String> = name
        .split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            KEY_ALIASES
                .iter()
                .find(|(alias, _)| *alias == p.to_lowercase())
                .map_or_else(|| p.to_string(), |(_, sym)| sym.to_string())
        })
        .collect();
    if parts.is_empty() {
        name.to_string()
    } else {
        parts.join("+")
    }
}

/// Keys as xdotool takes them: modifiers followed by one key become one
/// chord (`["ctrl", "l"]` is `ctrl+l`).
pub fn key_sequence(keys: &[String]) -> Vec<String> {
    let mapped: Vec<String> = keys
        .iter()
        .filter(|k| !k.is_empty())
        .map(|k| key(k))
        .collect();
    if mapped.len() <= 1 {
        return mapped;
    }
    let (last, modifiers) = mapped.split_last().expect("two or more keys");
    if modifiers
        .iter()
        .all(|m| ["ctrl", "alt", "shift", "super"].contains(&m.to_lowercase().as_str()))
    {
        vec![format!("{}+{last}", modifiers.join("+"))]
    } else {
        mapped
    }
}

fn modifier(m: Option<&str>) -> Option<&'static str> {
    match m?.trim().to_lowercase().as_str() {
        "shift" => Some("shift"),
        "ctrl" | "control" => Some("ctrl"),
        "alt" => Some("alt"),
        "super" | "meta" | "cmd" | "command" => Some("super"),
        _ => None,
    }
}

fn button(b: Option<&str>) -> &'static str {
    match b {
        Some("right") => "3",
        Some("middle") => "2",
        _ => "1",
    }
}

/// The xdotool argument lists for an action (Harbor's `build_xdotool_argv`),
/// or `None` for actions xdotool does not perform.
pub fn xdotool_args(a: &Action, width: u32, height: u32) -> Option<Vec<Vec<String>>> {
    let s = |v: i64| v.to_string();
    let x = s(a.x.unwrap_or(0));
    let y = s(a.y.unwrap_or(0));
    let m = modifier(a.modifier.as_deref());
    let click = |code: &str, repeat: u32| {
        let mut argv = vec!["mousemove".to_string(), x.clone(), y.clone()];
        if let Some(m) = m {
            argv.extend(["keydown".into(), m.into()]);
        }
        if repeat > 1 {
            argv.extend([
                "click".into(),
                "--repeat".into(),
                repeat.to_string(),
                code.into(),
            ]);
        } else {
            argv.extend(["click".into(), code.into()]);
        }
        if let Some(m) = m {
            argv.extend(["keyup".into(), m.into()]);
        }
        argv
    };
    let v = |items: &[&str]| items.iter().map(|i| i.to_string()).collect::<Vec<_>>();
    Some(match a.kind.as_str() {
        "click" => vec![click(button(a.button.as_deref()), 1)],
        "double_click" => vec![click("1", 2)],
        "triple_click" => vec![click("1", 3)],
        "right_click" => vec![click("3", 1)],
        "middle_click" => vec![click("2", 1)],
        "mouse_down" => vec![v(&["mousemove", &x, &y, "mousedown", "1"])],
        "mouse_up" => vec![v(&["mousemove", &x, &y, "mouseup", "1"])],
        "mouse_move" => vec![v(&["mousemove", &x, &y])],
        "type" => vec![v(&[
            "type",
            "--clearmodifiers",
            "--",
            a.text.as_deref().unwrap_or(""),
        ])],
        "key" | "keypress" => key_sequence(a.keys.as_deref().unwrap_or_default())
            .into_iter()
            .map(|k| v(&["key", "--clearmodifiers", &k]))
            .collect(),
        "drag" => {
            let ex = s(a.end_x.or(a.x).unwrap_or(0));
            let ey = s(a.end_y.or(a.y).unwrap_or(0));
            vec![v(&[
                "mousemove",
                &x,
                &y,
                "mousedown",
                "1",
                "mousemove",
                &ex,
                &ey,
                "mouseup",
                "1",
            ])]
        }
        "scroll" => {
            let cx = s(a.x.unwrap_or(i64::from(width / 2)));
            let cy = s(a.y.unwrap_or(i64::from(height / 2)));
            let sy = a.scroll_y.unwrap_or(500);
            let sx = a.scroll_x.unwrap_or(0);
            let mut argv = v(&["mousemove", &cx, &cy]);
            if let Some(m) = m {
                argv.extend(["keydown".into(), m.into()]);
            }
            if sy != 0 {
                let clicks = (sy.abs() / 100).max(1).to_string();
                let btn = if sy > 0 { "5" } else { "4" };
                argv.extend(v(&["click", "--repeat", &clicks, btn]));
            }
            if sx != 0 {
                let clicks = (sx.abs() / 100).max(1).to_string();
                let btn = if sx > 0 { "7" } else { "6" };
                argv.extend(v(&["click", "--repeat", &clicks, btn]));
            }
            if let Some(m) = m {
                argv.extend(["keyup".into(), m.into()]);
            }
            vec![argv]
        }
        _ => return None,
    })
}

/// `DISPLAY=:1 xdotool <args>`.
fn xdotool(args: &[String]) -> String {
    let words: Vec<String> = std::iter::once("xdotool".to_string())
        .chain(args.iter().map(|a| q(a)))
        .collect();
    format!("DISPLAY={DISPLAY} {}", words.join(" "))
}

/// A running desktop.
pub struct Desktop<'a> {
    pub sandbox: &'a dyn Sandbox,
    pub user: &'a str,
    pub env: &'a BTreeMap<String, String>,
    pub width: u32,
    pub height: u32,
    /// The next screenshot is cropped to this region, then it resets.
    zoom: Option<[i64; 4]>,
}

/// Why the desktop could not start.
#[derive(Debug)]
pub enum StartError {
    /// The image lacks the desktop; `missing` names what.
    NoDisplay {
        missing: String,
    },
    /// A program started but the display never came up; the log's tail.
    NotReady(String),
    Sandbox(SandboxError),
}

impl From<SandboxError> for StartError {
    fn from(e: SandboxError) -> Self {
        StartError::Sandbox(e)
    }
}

impl<'a> Desktop<'a> {
    pub fn new(
        sandbox: &'a dyn Sandbox,
        user: &'a str,
        env: &'a BTreeMap<String, String>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            sandbox,
            user,
            env,
            width,
            height,
            zoom: None,
        }
    }

    async fn sh(&self, command: String, timeout: Duration) -> Result<ExecOutput, SandboxError> {
        let mut env = self.env.clone();
        env.insert("DISPLAY".into(), DISPLAY.into());
        self.sandbox
            .exec(
                Exec::new(command)
                    .user(self.user.to_string())
                    .cwd("/tmp".to_string())
                    .envs(env)
                    .timeout(timeout),
            )
            .await
    }

    /// Check the image has a desktop, then start the display, a window
    /// manager (when the image has one), and a browser (when it has one),
    /// as Harbor's session does. Returns the browser's path, if any.
    pub async fn start(&self) -> Result<Option<String>, StartError> {
        let probe = self
            .sh(
                "m=; command -v Xvfb >/dev/null 2>&1 || m=\"$m Xvfb\"; \
                 command -v xdotool >/dev/null 2>&1 || m=\"$m xdotool\"; \
                 { command -v import || command -v scrot; } >/dev/null 2>&1 || m=\"$m import/scrot\"; \
                 echo \"BF missing$m\"; \
                 for b in chromium chromium-browser google-chrome google-chrome-stable firefox-esr firefox; do \
                   p=$(command -v $b 2>/dev/null) && { echo \"BF browser $p\"; break; }; done"
                    .to_string(),
                Duration::from_secs(30),
            )
            .await?;
        let missing = probe
            .stdout
            .lines()
            .find_map(|l| l.strip_prefix("BF missing"))
            .unwrap_or(" (no answer)")
            .trim()
            .to_string();
        if !missing.is_empty() {
            return Err(StartError::NoDisplay { missing });
        }
        let browser = probe
            .stdout
            .lines()
            .find_map(|l| l.strip_prefix("BF browser "))
            .map(str::to_string);
        let (w, h) = (self.width, self.height);
        let start = format!(
            "mkdir -p {RUNTIME_DIR}/profile; \
             if ! test -S /tmp/.X11-unix/X1; then \
               setsid nohup Xvfb {DISPLAY} -screen 0 {w}x{h}x24 -nolisten tcp >> {RUNTIME_DIR}/xvfb.log 2>&1 < /dev/null & \
             fi; \
             i=0; while ! test -S /tmp/.X11-unix/X1 && [ $i -lt 100 ]; do sleep 0.2; i=$((i+1)); done; \
             test -S /tmp/.X11-unix/X1 || {{ echo 'BF no-display'; tail -n 20 {RUNTIME_DIR}/xvfb.log; exit 3; }}; \
             if command -v startxfce4 >/dev/null 2>&1; then \
               setsid nohup startxfce4 >> {RUNTIME_DIR}/wm.log 2>&1 < /dev/null & sleep 2; pkill -f xfce4-panel >/dev/null 2>&1; \
             else for wm in xfwm4 openbox fluxbox matchbox-window-manager; do \
               command -v $wm >/dev/null 2>&1 && {{ setsid nohup $wm >> {RUNTIME_DIR}/wm.log 2>&1 < /dev/null & break; }}; done; fi; \
             echo 'BF display ready'"
        );
        let out = self.sh(start, Duration::from_secs(60)).await?;
        if !out.stdout.contains("BF display ready") {
            return Err(StartError::NotReady(out.combined()));
        }
        if let Some(b) = &browser {
            self.start_browser(b).await?;
        }
        Ok(browser)
    }

    /// Start the browser filling the display, and wait for its window.
    async fn start_browser(&self, browser: &str) -> Result<(), StartError> {
        let (w, h) = (self.width, self.height);
        let args = if browser.contains("firefox") {
            format!("--width {w} --height {h} about:blank")
        } else {
            // Harbor's Chromium flags, without remote debugging.
            format!(
                "--no-sandbox --disable-dev-shm-usage --disable-gpu --ignore-certificate-errors \
                 --user-data-dir={RUNTIME_DIR}/profile --window-position=0,0 --window-size={w},{h} \
                 --no-first-run --no-default-browser-check --disable-default-apps --disable-dev-tools \
                 --disable-extensions --disable-infobars --disable-features=Translate,HttpsUpgrades \
                 --test-type --lang=en-US about:blank"
            )
        };
        let start = format!(
            "setsid nohup {} {args} >> {RUNTIME_DIR}/browser.log 2>&1 < /dev/null & \
             i=0; while [ $i -lt 120 ]; do \
               xdotool search --onlyvisible --class 'chrom|firefox' >/dev/null 2>&1 && {{ echo 'BF browser ready'; break; }}; \
               sleep 0.5; i=$((i+1)); done; \
             wid=$(xdotool search --onlyvisible --class 'chrom|firefox' 2>/dev/null | tail -n 1); \
             [ -n \"$wid\" ] && {{ xdotool windowmove \"$wid\" 0 0; xdotool windowsize \"$wid\" {w} {h}; xdotool windowactivate \"$wid\"; }} >/dev/null 2>&1; true",
            q(browser)
        );
        let out = self.sh(start, Duration::from_secs(90)).await?;
        if !out.stdout.contains("BF browser ready") {
            let log = self
                .sh(
                    format!("tail -n 30 {RUNTIME_DIR}/browser.log 2>/dev/null"),
                    Duration::from_secs(10),
                )
                .await
                .map(|o| o.stdout)
                .unwrap_or_default();
            return Err(StartError::NotReady(format!(
                "{browser} opened no window within 60s: {log}"
            )));
        }
        Ok(())
    }

    /// Take a screenshot of the whole display (cropped once after `zoom`),
    /// as PNG bytes.
    pub async fn screenshot(&mut self) -> Result<Vec<u8>, String> {
        let shot = format!("{RUNTIME_DIR}/shot.png");
        let crop = match self.zoom.take() {
            Some([x0, y0, x1, y1]) => {
                let (w, h) = ((x1 - x0).max(1), (y1 - y0).max(1));
                format!(" && convert {shot} -crop {w}x{h}+{x0}+{y0} +repage {shot}")
            }
            None => String::new(),
        };
        let command = format!(
            "{{ import -window root {shot} 2>/dev/null || scrot -o {shot}; }}{crop} && base64 -w0 {shot}"
        );
        let out = self
            .sh(command, Duration::from_secs(60))
            .await
            .map_err(|e| e.to_string())?;
        if !out.success() {
            return Err(format!(
                "the screenshot failed (exit {}): {}",
                out.exit_code,
                out.stderr.trim()
            ));
        }
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(out.stdout.trim())
            .map_err(|e| format!("the screenshot was not base64 ({e})"))
    }

    /// Whether the display is still up.
    pub async fn alive(&self) -> bool {
        self.sh(
            "test -S /tmp/.X11-unix/X1".to_string(),
            Duration::from_secs(10),
        )
        .await
        .is_ok_and(|o| o.success())
    }

    /// Perform one action. Sandbox failures are `Err`; an action the model
    /// got wrong is an `Ok` outcome it reads.
    pub async fn act(&mut self, a: &Action) -> Result<Outcome, SandboxError> {
        let timeout = Duration::from_secs(60);
        if matches!(a.kind.as_str(), "key" | "keypress")
            && let Some(keys) = &a.keys
        {
            let combo = keys
                .iter()
                .map(|k| k.trim().to_lowercase())
                .filter(|k| !k.is_empty())
                .collect::<Vec<_>>()
                .join("+");
            if BLOCKED_COMBOS.contains(&combo.as_str()) {
                return Ok(Outcome::blocked(
                    "Action blocked: developer tools are not available in this environment.",
                ));
            }
        }
        if a.kind == "type"
            && a.text
                .as_deref()
                .is_some_and(|t| t.to_lowercase().contains("view-source:"))
        {
            return Ok(Outcome::blocked(
                "Action blocked: view-source is not available in this environment.",
            ));
        }
        if a.kind == "navigate"
            && let Some(url) = &a.url
            && BLOCKED_SCHEMES
                .iter()
                .any(|s| url.to_lowercase().starts_with(s))
        {
            return Ok(Outcome::blocked(
                "Action blocked: this URL scheme is not available in this environment.",
            ));
        }
        match a.kind.as_str() {
            "wait" => {
                tokio::time::sleep(Duration::from_secs(1)).await;
                return Ok(Outcome::ok());
            }
            "screenshot" => return Ok(Outcome::ok()),
            k if TERMINAL.contains(&k) => {
                return Ok(Outcome {
                    status: "done",
                    text: String::new(),
                });
            }
            "zoom" => {
                self.zoom = a.zoom_region;
                return Ok(Outcome::ok());
            }
            "hold_key" => {
                let keys = key_sequence(a.keys.as_deref().unwrap_or_default());
                let secs = a.duration.unwrap_or(1.0).clamp(0.0, 10.0);
                let down: Vec<String> = keys
                    .iter()
                    .map(|k| xdotool(&["keydown".into(), k.clone()]))
                    .collect();
                let up: Vec<String> = keys
                    .iter()
                    .rev()
                    .map(|k| xdotool(&["keyup".into(), k.clone()]))
                    .collect();
                let command = format!("{}; sleep {secs}; {}", down.join("; "), up.join("; "));
                return self.run(command, timeout).await;
            }
            "navigate" => {
                let url = a.url.as_deref().unwrap_or("about:blank");
                let steps = [
                    xdotool(&["key".into(), "--clearmodifiers".into(), "ctrl+l".into()]),
                    xdotool(&["key".into(), "--clearmodifiers".into(), "ctrl+a".into()]),
                    xdotool(&[
                        "type".into(),
                        "--clearmodifiers".into(),
                        "--".into(),
                        url.into(),
                    ]),
                    xdotool(&["key".into(), "--clearmodifiers".into(), "Return".into()]),
                ];
                let command = format!("{} && sleep 1", steps.join(" && sleep 0.2 && "));
                return self.run(command, timeout).await;
            }
            _ => {}
        }
        match xdotool_args(a, self.width, self.height) {
            None => Ok(Outcome::error(format!(
                "Unsupported action type: {}",
                a.kind
            ))),
            Some(argvs) => {
                let command = argvs
                    .iter()
                    .map(|argv| xdotool(argv))
                    .collect::<Vec<_>>()
                    .join(" && ");
                self.run(command, timeout).await
            }
        }
    }

    async fn run(&self, command: String, timeout: Duration) -> Result<Outcome, SandboxError> {
        let out = self.sh(command, timeout).await?;
        if out.timed_out {
            return Ok(Outcome::error(format!(
                "The action timed out after {} seconds. The computer may still be processing it.",
                timeout.as_secs()
            )));
        }
        if out.success() {
            Ok(Outcome::ok())
        } else {
            Ok(Outcome::error(format!(
                "The action failed (exit {}): {}",
                out.exit_code,
                out.combined().trim()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action(kind: &str) -> Action {
        Action {
            kind: kind.into(),
            ..Action::default()
        }
    }

    #[test]
    fn keys_follow_harbors_aliases_and_chords() {
        let k = |v: &[&str]| key_sequence(&v.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(k(&["ctrl", "l"]), ["ctrl+l"]);
        assert_eq!(k(&["Enter"]), ["Return"]);
        assert_eq!(k(&["cmd+shift+t"]), ["super+shift+t"]);
        assert_eq!(k(&["a", "b"]), ["a", "b"]);
    }

    #[test]
    fn clicks_scrolls_and_drags_become_harbors_xdotool_lines() {
        let mut a = action("click");
        a.x = Some(10);
        a.y = Some(20);
        a.modifier = Some("Control".into());
        assert_eq!(
            xdotool_args(&a, 1024, 900).unwrap(),
            [[
                "mousemove",
                "10",
                "20",
                "keydown",
                "ctrl",
                "click",
                "1",
                "keyup",
                "ctrl"
            ]]
        );
        let mut a = action("scroll");
        a.scroll_y = Some(-250);
        assert_eq!(
            xdotool_args(&a, 1024, 900).unwrap(),
            [["mousemove", "512", "450", "click", "--repeat", "2", "4"]]
        );
        let mut a = action("drag");
        a.x = Some(1);
        a.y = Some(2);
        a.end_x = Some(3);
        a.end_y = Some(4);
        assert_eq!(
            xdotool_args(&a, 1024, 900).unwrap()[0].join(" "),
            "mousemove 1 2 mousedown 1 mousemove 3 4 mouseup 1"
        );
        assert!(xdotool_args(&action("navigate"), 1024, 900).is_none());
        assert_eq!(
            xdotool(&["type".into(), "--".into(), "it's".into()]),
            "DISPLAY=:1 xdotool type -- 'it'\"'\"'s'"
        );
    }
}
