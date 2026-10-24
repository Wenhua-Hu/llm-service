# Agent module

`runner.py` implements the `/agent/run` workflow: a planner call, the allowed
`lookup_order` tool, and a final answer call. The HTTP route stays in
`server/app.py`; it passes the authenticated request and service dependencies to
the runner.
