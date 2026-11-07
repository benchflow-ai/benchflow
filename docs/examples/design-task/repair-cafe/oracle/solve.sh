#!/bin/bash
# A plain reference page: it states every fact the brief asks for, with little design.
mkdir -p /app/out
cat > /app/out/index.html <<'HTML'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fix-It Friday</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 36rem; padding: 1.5rem; line-height: 1.5; }
  h1 { font-size: 1.8rem; }
</style>
</head>
<body>
<h1>Fix-It Friday: bring it broken, take it home fixed</h1>
<p>Volunteers help you repair your things for free.</p>
<h2>What to bring</h2>
<ul><li>A lamp that won't switch on</li><li>A bike with a flat or a loose chain</li><li>A jacket with a broken zip</li></ul>
<h2>When</h2>
<p>The first and third Friday of each month, 6 to 9 pm.</p>
<h2>Where</h2>
<p>The community hall, 40 Mill Street.</p>
</body>
</html>
HTML
