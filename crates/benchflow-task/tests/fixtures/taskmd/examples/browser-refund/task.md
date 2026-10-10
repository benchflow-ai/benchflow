Your order #4821 from the Elm Street Goods shop arrived with one item damaged: the blue ceramic mug. Request a refund for that item only, and say in the request that it arrived damaged. If the store says the refund window has passed, choose store credit instead. You are already signed in.

```toml task
name = "examples/browser-refund"
title = "Request a refund for one damaged item"
version = "3.2.0"
keywords = ["computer-use", "browser", "customer-service"]

[agent]
timeout = "10m"

[world]
kind = "virtual"
profile = "bf-browser@1"
interface = "computer@1"

[world.browser]
start_url = "http://shop.local/orders/4821"
viewport = "1440x900"
services = ["world/shop"]
storage_state = "world/session.json"
locale = "en_US"
timezone = "America/New_York"

[world.record]
streams = ["video", "har", "service-db-snapshot"]
required = ["har", "service-db-snapshot"]
```
