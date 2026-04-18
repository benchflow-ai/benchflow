#!/bin/bash
cat > /app/review.md << 'EOF'
# Code Review: user_service.py

- BUG: SQL injection vulnerability in `find_user()` (line 14). User input is directly interpolated into the SQL query using an f-string: `f"SELECT * FROM users WHERE username = '{username}'"`. An attacker can inject arbitrary SQL by providing a username like `' OR '1'='1`. Fix: use parameterized queries with `?` placeholders, like the `create_user()` function already does correctly.

- NOTE: `authenticate()` compares passwords in plaintext. Should use hashed passwords with bcrypt or argon2. Not a SQL injection but a security concern.
EOF
