Design and build a single-page website for Oda Clay, a two-person ceramics studio in Portland. It should show their three product lines, their weekend class schedule, and a way to join the mailing list. The owners want it to feel calm and handmade, not like a template. It has to work on a phone as well as on a laptop.

Build it as static HTML, CSS, and JavaScript in `/work/site/`. Use the photos in `/assets/`; do not use stock imagery.

```toml task
name = "examples/ceramics-landing"
title = "A landing page for a ceramics studio"
version = "2.2.0"
keywords = ["web-design", "taste", "frontend"]

[agent]
timeout = "1h"

[sandbox]
image = "ghcr.io/example/web-static:2026.09"

[preference]
protocol = "pairwise"
compare = "same-task-version"
blind = true
randomize_sides = true
answers = ["A", "B", "tie", "cannot-judge"]
directions = ["overall", "aesthetic", "fit-to-brief"]
raters_per_pair = 3
rater_pool = "designers"
gold_pairs = "evidence/gold-pairs.jsonl"
min_agreement = { metric = "krippendorff_alpha", value = 0.4 }
aggregate = "davidson-bradley-terry"
export_pairs = true
```
