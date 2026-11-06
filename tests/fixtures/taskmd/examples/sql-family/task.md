The SQLite database at `/data/shop.db` has three tables: `customers(id, name, country)`, `orders(id, customer_id, placed_at, status)`, and `order_items(order_id, product, quantity, unit_price)`. `placed_at` is a date written as `YYYY-MM-DD`.

Write one SQL query that returns the top {{k}} customers by {{metric}} for completed orders placed in {{year}}. Revenue is the sum of `quantity * unit_price` over a customer's order items, order_count is the number of the customer's orders, and items_bought is the sum of `quantity`. Count only orders whose `status` is `completed`, and leave out customers with no such orders. Return two columns, `customer_id` and `{{metric}}`, highest first, breaking ties by customer id, lowest first.

Save the query to `/work/answer.sql`. It will be run against another database with the same tables, so it must not depend on this database's rows.

```toml task
name = "examples/sql-family"
title = "Top-k customers, a seeded task family"
version = "3.1.0"
keywords = ["sql", "rl", "task-family"]

[agent]
timeout = "10m"
budget = { tool_calls = 20, tokens = "400K" }

[sandbox]
cpus = 1
memory = "1 GB"
network = "none"
workdir = "/work"

[verifier]
timeout = "1m"

[family]
generator = "family/generate.py"
params = { k = [3, 5, 10], metric = ["revenue", "order_count", "items_bought"], year = [2023, 2024, 2025] }

[family.splits]
train = { role = "train", rule = "seed % 10 < 8" }
validation = { role = "validation", rule = "seed % 10 == 8" }
test = { role = "test", rule = "seed % 10 == 9" }

[training]
reward = "strict"
difficulty = "evidence/difficulty.json"
```
