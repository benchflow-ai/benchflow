---
verifier:
  strategies:
    judge:
      type: agent-judge
      role: page-judge
      isolation: verifier-only
      model: openai/mimo-v2.6-flash
      inputs: [/app/out/index.html]
---

MiMo V2.6 Flash reads the HTML of `/app/out/index.html` and checks it against
three pass/fail criteria. The reward is 1 only when all three pass.

## role:page-judge

You grade one web page, `/app/out/index.html`, from its HTML source. You can't
see it rendered. Give each criterion "pass" or "fail".

- facts: pass when the page states every fact in the brief, each correctly: what
  happens (Fix-It Friday, a free repair café run by volunteers), the three things
  to bring (a lamp, a bike, a jacket with a broken zip), when (the first and
  third Friday of each month, 6 to 9 pm) and where (the community hall, 40 Mill
  Street). Fail when any of them is missing or different.
- first_screen: pass when the head has a viewport meta tag with
  width=device-width and, in the order of the body, when (the days and the
  hours) and where (the address) come after no more than the headline, a short
  paragraph and a short list. Fail when longer content comes first, such as a
  description of each thing to bring.
- self_contained: pass when the page is one complete HTML document that works on
  its own: its CSS is inside it, it loads no other file by a relative path, and
  it has no placeholder or unfinished text, such as lorem ipsum, TODO or an empty
  section. Links within the page and inline SVG are fine. Fail otherwise.

In the JSON you return, "score" is 1 when all three pass and 0 otherwise, and
"reasoning" gives each criterion in the form "name: pass, reason" or
"name: fail, reason".
