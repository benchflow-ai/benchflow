I just made some changes to my personal site and checked out master, but now I can't find those changes. Please help me find them and merge them into master.

```toml task
[about]
author_name = "TheMikeMerrill"
author_email = "mchlmerrill@gmail.com"
difficulty = "easy"
category = "software-engineering"
tags = ["coding", "version-control"]
expert_time_estimate_min = 5.0
junior_time_estimate_min = 20.0

[agent]
timeout = "15m"

[sandbox]
image = "alexgshaw/fix-git:20251031"
cpus = 1
memory = "2 GB"
disk = "10 GB"
build_timeout = "10m"

[verifier]
timeout = "15m"
mount = "/tests"

[oracle]
mount = "/solution"

[import.harbor]
version = "1.0"
```
