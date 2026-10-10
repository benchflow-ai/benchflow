Fix it.

```stage a
A.
```

```stage b
B.
```

```stage c
C.
```

```toml task
name = "acme/stage-unlock-forms"
version = "1.0.0"
[stages.a]
unlock = "at_turn:3"

[stages.b]
unlock = "on_request"

[stages.c]
unlock = "at_turn:x"
```
