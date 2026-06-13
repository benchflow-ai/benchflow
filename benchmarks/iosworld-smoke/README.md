# iOSWorld Smoke

This fixture is an iOSWorld-shaped repository used to verify inbound adapter
recognition and provider-honest unsupported reporting.

It is not runnable on Docker, Daytona, Modal, or Cua. A runnable iOSWorld slice
requires a macOS/iOS Simulator provider with Xcode, an iOS Simulator runtime,
Appium/XCUITest, and iOSWorld app bootstrap support.

Expected checks:

```bash
uv run bench tasks check benchmarks/iosworld-smoke/repo \
  --level runtime-capability \
  --sandbox cua \
  --json

uv run bench environment check benchmarks/iosworld-smoke/repo \
  --sandbox cua \
  --json
```

Both commands should exit non-zero with `status=unsupported-adapter-task`,
`adapter=iosworld`, and `details.required_provider=macos-ios-simulator`.
