# Native system boundary

AEGIS keeps policy, approvals, path validation, model orchestration, and
structured results in Python. OS-sensitive process lifecycle work is delegated
to the small C++17 helper in [`native/aegis_exec.cpp`](../native/aegis_exec.cpp)
when it is built.

The helper owns:

- creation of a fresh process group (POSIX: setpgid, Windows: Job Object);
- bounded waiting; and
- terminating the complete child process group on timeout.

Docker and Bubblewrap are already native system components, so AEGIS invokes
their supported CLIs instead of duplicating their runtimes in C++. Cgroup v2
configuration remains explicit and auditable in Python because it is a small
filesystem control-plane operation, while the sandbox process lifecycle uses
the native helper.

Build manually:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
mkdir -p native/bin
cp native/build/aegis-exec native/bin/aegis-exec
```

On Windows, the helper uses Job Objects to achieve process-group isolation and
timeout termination. The Windows binary is `native/bin/aegis-exec.exe` and is
selected automatically by the Python adapter. On POSIX systems, it uses
`setpgid` and signal forwarding. The Windows branch still needs validation on
an actual Windows host because Linux CI cannot compile or execute it.
