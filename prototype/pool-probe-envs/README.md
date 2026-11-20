# Pool probe across pixi envs (throwaway)

Wraps `cuda::shared_pinned_memory_pool` in a `cuda::mr::shared_resource` handle, converts it to
`rmm::host_device_async_resource_ref`, and exercises limits, bad nodes, zero bytes, alignment,
`trim_to`, attributes and both device-access narrowing routes. One forked child per case.

Run from the repo root, once per env:

    pixi run -e <env> bash prototype/pool-probe-envs/run.sh <env>

Results: `out-<env>.txt` (`default` is cuda-13-nightly). Machine: 2x RTX PRO 6000 Blackwell
(sm_120) + RTX A400 (sm_86), one NUMA node, driver 13.4 in every env. Never merged.
