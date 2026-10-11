//! The HTML page: 0.8's template with every asset inlined, plus the boot
//! data its scripts start from (0.8's `viewer/render.py`).
//!
//! The page is self-contained (styles, fonts, scripts, and data all
//! inline), so a rendered trial works as a file opened from disk. Boot data
//! is either `{"mode": "browse", "capped", "rollouts"}` (the run list, which
//! loads each run from `/api/rollout`) or `{"mode": "single", "payload",
//! "branches"?}` (one trial with its branch children embedded).

use std::path::Path;
use std::sync::OnceLock;

use crate::catalog::{Run, is_rollout_dir, rollout_summary};
use crate::payload::{branch_child_payloads, build_payload};
use crate::py;
use crate::pyjson::{Map, Value, Writer, object};
use crate::target::ViewError;

/// 0.8's assets, embedded unchanged.
mod assets {
    pub const TEMPLATE: &str = include_str!("../assets/template.html");
    pub const THEME_CSS: &str = include_str!("../assets/theme.css");
    pub const VIEWER_CSS: &str = include_str!("../assets/viewer.css");
    /// Third-party scripts (see `assets/vendor/LICENSE.highlight.js`).
    pub const VENDOR_JS: [&str; 1] = [include_str!("../assets/vendor/highlight.min.js")];
    /// The app's scripts, in load order: `viewer.js` declares the `BF`
    /// namespace the others extend.
    pub const SCRIPTS: [&str; 5] = [
        include_str!("../assets/viewer.js"),
        include_str!("../assets/theme.js"),
        include_str!("../assets/detail.js"),
        include_str!("../assets/catalog.js"),
        include_str!("../assets/boot.js"),
    ];
}

/// The compare, outcomes, and Pareto pages (new in the Rust viewer): their own
/// template, with 0.8's theme, styles, and shared scripts.
mod view_assets {
    pub const TEMPLATE: &str = include_str!("../assets/views.html");
    pub const VIEWS_CSS: &str = include_str!("../assets/views.css");
    /// `viewer.js` declares `BF` and its helpers, `theme.js` the toggle.
    pub const SCRIPTS: [&str; 3] = [
        include_str!("../assets/viewer.js"),
        include_str!("../assets/theme.js"),
        include_str!("../assets/views.js"),
    ];
}

const TITLE_MARKER: &str = "__BENCHFLOW_TITLE__";
const APP_MARKER: &str = "__BENCHFLOW_APP__";
const NAV_MARKER: &str = "<!--__BENCHFLOW_NAV__-->";
const PAYLOAD_MARKER: &str = "__BENCHFLOW_PAYLOAD__";

/// The assembled template, split around the title and the boot data.
struct Template {
    /// Everything before the title.
    head: String,
    /// From after the title to the boot data.
    middle: String,
    /// Everything after the boot data.
    tail: String,
}

/// Python's text mode turns `\r\n` and `\r` into `\n` when 0.8 reads an asset.
fn text_mode(asset: &str) -> String {
    crate::values::decode_text(asset.as_bytes())
}

/// The template with the styles and scripts put in, as 0.8's
/// `_load_template` does it (each marker's first occurrence replaced).
pub(crate) fn assembled_template() -> String {
    let scripts: Vec<String> = assets::SCRIPTS
        .iter()
        .map(|script| text_mode(script))
        .collect();
    let vendor: Vec<String> = assets::VENDOR_JS
        .iter()
        .map(|script| text_mode(script))
        .collect();
    text_mode(assets::TEMPLATE)
        .replacen(
            "/*__BENCHFLOW_THEME_CSS__*/",
            &text_mode(assets::THEME_CSS),
            1,
        )
        .replacen(
            "/*__BENCHFLOW_VIEWER_CSS__*/",
            &text_mode(assets::VIEWER_CSS),
            1,
        )
        .replacen("//__BENCHFLOW_VENDOR_JS__", &vendor.join("\n\n"), 1)
        .replacen("//__BENCHFLOW_VIEWER_JS__", &scripts.join("\n\n"), 1)
}

fn template() -> &'static Template {
    static TEMPLATE: OnceLock<Template> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let page = assembled_template();
        // 0.8 replaces the boot-data marker first, then the first title
        // marker; the `<title>` marker comes before the boot data, so the
        // page splits cleanly at both.
        let title = page
            .find(TITLE_MARKER)
            .expect("the template has a title marker");
        let payload = page
            .find(PAYLOAD_MARKER)
            .expect("the template has a boot-data marker");
        assert!(title < payload, "the title marker precedes the boot data");
        Template {
            head: page[..title].to_string(),
            middle: page[title + TITLE_MARKER.len()..payload].to_string(),
            tail: page[payload + PAYLOAD_MARKER.len()..].to_string(),
        }
    })
}

/// Write a page with `title` and `boot` data (0.8's `_render_shell`). The
/// boot JSON is escaped for an HTML `<script>` element, so trajectory text
/// can never close it early.
pub fn write_page(writer: &mut Writer<'_>, title: &str, boot: &Value) {
    let template = template();
    writer.raw(&template.head);
    writer.raw(&py::html_escape(title));
    writer.raw(&template.middle);
    writer.set_script_safe(true);
    writer.value(boot);
    writer.set_script_safe(false);
    writer.raw(&template.tail);
}

/// Where the run list page gets its one addition, the links to the other
/// views: just before the theme button in the wordmark row.
const NAV_ANCHOR: &str = "<button id=\"theme-toggle\"";

/// The links between the run list, compare, outcomes, and Pareto views, with
/// `current` (`runs`, `compare`, `outcomes`, or `pareto`) marked. Styled inline, so
/// the run list page needs no other change.
pub fn nav(current: &str) -> String {
    let link = |href: &str, name: &str| {
        let (mark, look) = if name == current {
            (
                " aria-current=\"page\"",
                "color:var(--foreground);background:var(--secondary);border-color:var(--border)",
            )
        } else {
            ("", "color:var(--muted-foreground)")
        };
        format!(
            "<a href=\"{href}\"{mark} style=\"padding:2px 10px;border:1px solid transparent;\
             border-radius:999px;text-decoration:none;{look}\">{name}</a>"
        )
    };
    format!(
        "<nav id=\"bf-views\" aria-label=\"views\" style=\"display:flex;flex-wrap:wrap;gap:2px;\
         margin-left:6px;font-size:12px\">{}{}{}{}</nav>",
        link("/", "runs"),
        link("/compare", "compare"),
        link("/outcomes", "outcomes"),
        link("/pareto", "pareto")
    )
}

/// The run list page a server sends: 0.8's page with [`nav`] added before
/// the theme button, and nothing else changed.
pub fn write_browse_page(writer: &mut Writer<'_>, title: &str, boot: &Value) {
    static MIDDLE: OnceLock<String> = OnceLock::new();
    let template = template();
    let middle = MIDDLE.get_or_init(|| {
        let at = template
            .middle
            .find(NAV_ANCHOR)
            .expect("the template has a theme button");
        format!(
            "{}{}\n      {}",
            &template.middle[..at],
            nav("runs"),
            &template.middle[at..]
        )
    });
    writer.raw(&template.head);
    writer.raw(&py::html_escape(title));
    writer.raw(middle);
    writer.set_script_safe(true);
    writer.value(boot);
    writer.set_script_safe(false);
    writer.raw(&template.tail);
}

/// The compare, outcomes, and Pareto template with its styles and scripts put in.
fn view_template() -> &'static str {
    static TEMPLATE: OnceLock<String> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let scripts: Vec<String> = view_assets::SCRIPTS
            .iter()
            .map(|script| text_mode(script))
            .collect();
        let css = [
            assets::THEME_CSS,
            assets::VIEWER_CSS,
            view_assets::VIEWS_CSS,
        ]
        .map(text_mode)
        .join("\n");
        text_mode(view_assets::TEMPLATE)
            .replacen("/*__BENCHFLOW_CSS__*/", &css, 1)
            .replacen("//__BENCHFLOW_JS__", &scripts.join("\n\n"), 1)
    })
}

/// Write a view page: `view` is `compare`, `outcomes`, or `pareto`,
/// `boot` its data (what `/api/<view>` answers for the same query).
pub fn write_view_page(writer: &mut Writer<'_>, view: &str, title: &str, boot: &Value) {
    let template = view_template();
    let mut rest = template;
    let label = match view {
        "compare" => "compare runs",
        "pareto" => "pareto",
        _ => "outcomes",
    };
    for (marker, fill) in [
        (TITLE_MARKER, py::html_escape(title)),
        (APP_MARKER, label.to_string()),
        (NAV_MARKER, nav(view)),
    ] {
        let at = rest
            .find(marker)
            .expect("the view template has its markers");
        writer.raw(&rest[..at]);
        writer.raw(&fill);
        rest = &rest[at + marker.len()..];
    }
    let at = rest
        .find(PAYLOAD_MARKER)
        .expect("the view template has a boot-data marker");
    writer.raw(&rest[..at]);
    writer.set_script_safe(true);
    writer.value(boot);
    writer.set_script_safe(false);
    writer.raw(&rest[at + PAYLOAD_MARKER.len()..]);
}

/// A page as a string.
pub fn render_page(title: &str, boot: &Value) -> String {
    let mut writer = Writer::new();
    write_page(&mut writer, title, boot);
    writer.into_string()
}

/// The run list's boot data: one row per run, plus whether the list was
/// capped. A trial a live `bench run` or `bench regrade` has not finished is
/// left out, as the other views and `bench results` leave it out: the run
/// list's stats (0.8's page, unchanged) would count it as unscored.
pub fn browse_boot(runs: &[Run], capped: bool) -> Value {
    object([
        ("mode", "browse".into()),
        ("capped", capped.into()),
        (
            "rollouts",
            Value::Array(
                runs.iter()
                    .filter(|run| !crate::views::in_progress(run))
                    .map(rollout_summary)
                    .collect(),
            ),
        ),
    ])
}

/// One trial's boot data, with every openable branch child embedded so the
/// Lineage tab's `?branch=` links work without a server.
pub fn single_boot(trial_dir: &Path) -> Value {
    let payload = build_payload(trial_dir);
    let branches: Map = branch_child_payloads(trial_dir, payload.lineage.as_ref());
    let mut boot = Map::new();
    boot.insert("mode".into(), "single".into());
    boot.insert("payload".into(), payload.into_value());
    if !branches.is_empty() {
        boot.insert("branches".into(), Value::Object(branches));
    }
    Value::Object(boot)
}

/// A trial as one self-contained HTML file (what 0.8 saves as
/// `trajectory.html`): open it from disk, no server needed.
///
/// `trial_dir` must be a trial folder, one that holds
/// `trajectory/acp_trajectory.jsonl`.
pub fn render_static(trial_dir: &Path) -> Result<String, ViewError> {
    let trial_dir = py::normalize(trial_dir);
    if !trial_dir.exists() {
        return Err(ViewError::NotFound(trial_dir));
    }
    if !trial_dir.is_dir() || !is_rollout_dir(&trial_dir) {
        return Err(ViewError::NotATrial(trial_dir));
    }
    let boot = single_boot(&trial_dir);
    Ok(render_page(&py::name_of(&trial_dir), &boot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_inlines_every_asset_in_order() {
        let page = assembled_template();
        assert!(!page.contains("/*__BENCHFLOW_THEME_CSS__*/"));
        assert!(!page.contains("//__BENCHFLOW_VIEWER_JS__"));
        let vendor = page.find("Highlight.js v11.9.0").unwrap();
        assert!(vendor < page.find("BF.highlight = ").unwrap());
        assert!(page.find("const BF = {}").unwrap() < page.find("BF.theme = ").unwrap());
        assert!(
            page.find("localStorage.getItem(\"bf-theme\")").unwrap()
                < page.find("--background:").unwrap()
        );
        assert!(!page.contains("cdnjs"));
    }

    #[test]
    fn hostile_text_cannot_leave_the_boot_script() {
        let hostile = "</script><script>alert(1)</script><!--<script>";
        let boot = object([
            ("mode", "single".into()),
            ("payload", object([("x", hostile.into())])),
        ]);
        let page = render_page("<b>&'\"", &boot);
        assert!(!page.contains(hostile));
        assert!(page.contains("\\u003c/script\\u003e"));
        assert!(page.contains("<title>&lt;b&gt;&amp;&#x27;&quot;</title>"));
    }

    #[test]
    fn a_title_equal_to_the_boot_marker_does_not_swallow_the_data() {
        let boot = object([
            ("mode", "browse".into()),
            ("capped", false.into()),
            ("rollouts", Value::Array(vec![])),
        ]);
        let page = render_page(PAYLOAD_MARKER, &boot);
        let data = page
            .split("type=\"application/json\">")
            .nth(1)
            .unwrap()
            .split("</script>")
            .next()
            .unwrap();
        assert_eq!(
            data,
            "{\"mode\": \"browse\", \"capped\": false, \"rollouts\": []}"
        );
    }

    #[test]
    fn the_run_list_page_only_gains_the_nav() {
        let boot = browse_boot(&[], false);
        let mut writer = Writer::new();
        write_browse_page(&mut writer, "jobs", &boot);
        let served = writer.into_string();
        let plain = render_page("jobs", &boot);
        let added = format!("{}\n      ", nav("runs"));
        assert_eq!(served.matches(&added).count(), 1);
        assert_eq!(served.replacen(&added, "", 1), plain);
        assert!(served.contains(&format!("{added}<button id=\"theme-toggle\"")));
    }

    #[test]
    fn view_pages_fill_every_marker_and_escape_the_title() {
        let boot = object([("mode", "compare".into()), ("x", "</script>".into())]);
        let mut writer = Writer::new();
        write_view_page(&mut writer, "compare", "<a>", &boot);
        let page = writer.into_string();
        for marker in [
            TITLE_MARKER,
            APP_MARKER,
            NAV_MARKER,
            PAYLOAD_MARKER,
            "__BENCHFLOW_CSS__",
            "__BENCHFLOW_JS__",
        ] {
            assert!(!page.contains(marker), "{marker}");
        }
        assert!(page.contains("<title>&lt;a&gt;</title>"));
        assert!(page.contains("<span class=\"app\">compare runs</span>"));
        assert!(page.contains("<a href=\"/compare\" aria-current=\"page\""));
        assert!(!page.contains("x\": \"</script>"));
        assert!(page.find("const BF = {}").unwrap() < page.find("BF.views = ").unwrap());
    }

    #[test]
    fn static_render_refuses_anything_but_a_trial_folder() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            render_static(dir.path()),
            Err(ViewError::NotATrial(_))
        ));
        std::fs::write(dir.path().join("session.jsonl"), "{}").unwrap();
        assert!(matches!(
            render_static(&dir.path().join("session.jsonl")),
            Err(ViewError::NotATrial(_))
        ));
        assert!(matches!(
            render_static(&dir.path().join("missing")),
            Err(ViewError::NotFound(_))
        ));
    }
}
