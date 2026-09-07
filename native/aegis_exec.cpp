// AEGIS native process boundary.
//
// Policy, approvals, path validation, and command parsing remain in Python.
// This small helper owns the OS-sensitive lifecycle: a fresh process group,
// bounded waiting, and group termination on timeout.  On Windows the helper
// uses Job Objects for similar functionality.

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#else
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
static int run_posix(const std::string& cwd, int timeout_ms,
                     const std::vector<std::string>& command) {
    if (command.empty()) return 64;
    pid_t child = fork();
    if (child < 0) return 70;
    if (child == 0) {
        if (!cwd.empty() && chdir(cwd.c_str()) != 0) _exit(126);
        setpgid(0, 0);
        std::vector<char*> argv;
        argv.reserve(command.size() + 1);
        for (const auto& item : command) argv.push_back(const_cast<char*>(item.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(errno == ENOENT ? 127 : 126);
    }
    setpgid(child, child);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    int status = 0;
    while (true) {
        pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            if (WIFEXITED(status)) return WEXITSTATUS(status);
            if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
            return 1;
        }
        if (result < 0 && errno != EINTR) return 70;
        if (std::chrono::steady_clock::now() >= deadline) {
            kill(-child, SIGTERM);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (waitpid(child, &status, WNOHANG) == 0) kill(-child, SIGKILL);
            waitpid(child, &status, 0);
            return 124;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
#endif

#ifdef _WIN32
static int run_windows(const std::string& cwd, int timeout_ms,
                       const std::vector<std::string>& command) {
    if (command.empty()) return 64;

    // Build the command line
    std::string cmdline;
    for (size_t i = 0; i < command.size(); ++i) {
        if (i > 0) cmdline += " ";
        // Simple quoting: if argument contains space, wrap in quotes
        if (command[i].find(' ') != std::string::npos) {
            cmdline += "\"";
            cmdline += command[i];
            cmdline += "\"";
        } else {
            cmdline += command[i];
        }
    }

    SECURITY_ATTRIBUTES saAttr = {};
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = nullptr;

    // Create pipes for child's STDOUT and STDERR
    HANDLE hChildStdoutRd = nullptr;
    HANDLE hChildStdoutWr = nullptr;
    HANDLE hChildStderrRd = nullptr;
    HANDLE hChildStderrWr = nullptr;

    if (!CreatePipe(&hChildStdoutRd, &hChildStdoutWr, &saAttr, 0) ||
        !SetHandleInformation(hChildStdoutRd, HANDLE_FLAG_INHERIT, 0) ||
        !CreatePipe(&hChildStderrRd, &hChildStderrWr, &saAttr, 0) ||
        !SetHandleInformation(hChildStderrRd, HANDLE_FLAG_INHERIT, 0)) {
        return 70;
    }

    STARTUPINFOW siW = {};
    siW.cb = sizeof(STARTUPINFOW);
    siW.hStdError = hChildStderrWr;
    siW.hStdOutput = hChildStdoutWr;
    siW.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    siW.dwFlags |= STARTF_USESTDHANDLES;

    // Convert command line to wide string
    int wlen = MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wcmdline(wlen);
    MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, wcmdline.data(), wlen);
    std::vector<wchar_t> wcwd;
    if (!cwd.empty()) {
        int cwd_len = MultiByteToWideChar(CP_UTF8, 0, cwd.c_str(), -1, nullptr, 0);
        wcwd.resize(cwd_len);
        MultiByteToWideChar(CP_UTF8, 0, cwd.c_str(), -1, wcwd.data(), cwd_len);
    }

    PROCESS_INFORMATION piProc = {};
    BOOL bSuccess = CreateProcessW(
        nullptr,           // application name
        wcmdline.data(),   // command line
        nullptr,           // process security attributes
        nullptr,           // thread security attributes
        TRUE,              // inherit handles
        CREATE_NEW_PROCESS_GROUP, // creation flags
        nullptr,           // environment
        wcwd.empty() ? nullptr : wcwd.data(), // current directory
        &siW,              // startup info
        &piProc);          // process info

    // Close pipe write handles (they are inherited by child)
    CloseHandle(hChildStdoutWr);
    CloseHandle(hChildStderrWr);

    if (!bSuccess) {
        CloseHandle(hChildStdoutRd);
        CloseHandle(hChildStderrRd);
        CloseHandle(piProc.hThread);
        CloseHandle(piProc.hProcess);
        return 70;
    }

    // Create a Job Object and set the kill on job close limit
    HANDLE hJob = CreateJobObjectW(nullptr, nullptr);
    if (hJob) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = {};
        jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli))) {
            CloseHandle(hJob);
            hJob = nullptr;
        } else {
            // Associate the process with the job
            AssignProcessToJobObject(hJob, piProc.hProcess);
        }
    }

    // Wait for the process to exit or timeout
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    DWORD ret = WaitForSingleObject(piProc.hProcess, timeout_ms);
    bool timed_out = false;
    if (ret == WAIT_TIMEOUT) {
        timed_out = true;
        // Terminate the job (which will kill the process and any children)
        if (hJob) {
            TerminateJobObject(hJob, 0);
        } else {
            TerminateProcess(piProc.hProcess, 0);
        }
        // Wait a bit for termination
        WaitForSingleObject(piProc.hProcess, 250);
    }

    // Get exit code
    DWORD exitCode = 0;
    if (!timed_out) {
        GetExitCodeProcess(piProc.hProcess, &exitCode);
    } else {
        exitCode = 124; // timeout
    }

    // Read output from pipes
    std::string stdout_str;
    std::string stderr_str;
    char buffer[4096];
    DWORD bytesRead;

    if (hJob) CloseHandle(hJob);
    CloseHandle(piProc.hThread);

    // Read stdout
    while (ReadFile(hChildStdoutRd, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        stdout_str.append(buffer, bytesRead);
    }
    CloseHandle(hChildStdoutRd);

    // Read stderr
    while (ReadFile(hChildStderrRd, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
        buffer[bytesRead] = '\0';
        stderr_str.append(buffer, bytesRead);
    }
    CloseHandle(hChildStderrRd);
    CloseHandle(piProc.hProcess);

    // Output to our stdout/stderr so the caller can capture
    if (!stdout_str.empty()) {
        fwrite(stdout_str.data(), 1, stdout_str.size(), stdout);
        fflush(stdout);
    }
    if (!stderr_str.empty()) {
        fwrite(stderr_str.data(), 1, stderr_str.size(), stderr);
        fflush(stderr);
    }

    return static_cast<int>(exitCode);
}
#endif

int main(int argc, char** argv) {
    std::string cwd;
    int timeout_ms = 120000;
    std::vector<std::string> command;
    bool after_separator = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--") { after_separator = true; continue; }
        if (after_separator) { command.push_back(arg); continue; }
        if (arg == "--cwd" && i + 1 < argc) { cwd = argv[++i]; continue; }
        if (arg == "--timeout-ms" && i + 1 < argc) { timeout_ms = std::atoi(argv[++i]); continue; }
        if (arg == "--version") { std::cout << "aegis-exec 1\n"; return 0; }
        return 64;
    }
    if (timeout_ms < 1 || timeout_ms > 120000) return 64;

#ifndef _WIN32
    return run_posix(cwd, timeout_ms, command);
#else
    return run_windows(cwd, timeout_ms, command);
#endif
}
