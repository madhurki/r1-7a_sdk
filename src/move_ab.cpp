// move_ab: slow, time-parameterised joint-space move A -> B (-> A ...) in ONE
// process, so FOC never drops between segments (no limp gaps, no sagging).
//
// Each segment is a quintic (minimum-jerk) profile: zero velocity and
// acceleration at both ends. Duration is chosen so the fastest joint never
// exceeds --vmax rad/s. Position + velocity feed-forward is sent every tick.
//
// Safety: targets are checked against URDF limits (minus a margin), tracking
// error above --max-err aborts, any motor fault/mode drop aborts, Ctrl-C
// aborts. On abort/finish the arm holds its measured pose briefly, then
// releases through damping only (kp=0) before mode-0 stop.

#include "arm_motor_bus.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

namespace {

using JointArray = std::array<float, kArmJointCount>;
using clock_t_ = std::chrono::steady_clock;

// URDF joint limits (rad), same as demo_arm_motion.cpp.
constexpr JointArray kLower = {-2.723f, 0.f, -0.349f, 0.f, -2.269f, -0.785f, -1.571f};
constexpr JointArray kUpper = {2.723f, 3.1416f, 3.491f, 3.1415f, 2.269f, 0.785f, 1.571f};

std::atomic<bool> g_running{true};
void onSignal(int) { g_running = false; }

struct Options {
    std::string port = "/dev/ttyUSB0";
    int baud = kArmDefaultBaudrate;
    int hand = 0;  // 1 right, 2 left
    bool a_current = false;
    bool a_set = false;
    JointArray a{};
    bool b_set = false;
    bool b_rel = false;
    JointArray b{};
    float vmax = 0.15f;      // rad/s, fastest joint
    float kp = 20.f;
    float kd = 1.0f;
    float max_err = 0.3f;    // rad
    float margin = 0.10f;    // rad inside URDF limits
    int cycles = 1;          // number of A->B->A round trips
    int dwell_ms = 1000;     // pause at each end
    int rate_hz = 100;
    bool hold_end = false;   // hold final pose until Ctrl-C
};

void usage(const char* a0) {
    std::cout
        << "Usage: " << a0 << " -r|-l --a cur|<7 rad> (--b <7 rad> | --b-rel <7 rad>) [opts]\n"
        << "  --port PATH       serial device (default /dev/ttyUSB0)\n"
        << "  --a cur           use the arm's current pose as A\n"
        << "  --a q0..q6        absolute A (URDF rad); arm first moves slowly to A\n"
        << "  --b q0..q6        absolute B\n"
        << "  --b-rel d0..d6    B = A + d (safest way to start)\n"
        << "  --vmax F          peak speed of fastest joint rad/s (default 0.15)\n"
        << "  --kp F --kd F     gains (default 20 / 1.0)\n"
        << "  --max-err F       abort if |q_ref - q_meas| > F rad (default 0.3)\n"
        << "  --margin F        stay F rad inside URDF limits (default 0.10)\n"
        << "  --cycles N        A->B->A round trips (default 1)\n"
        << "  --dwell-ms N      pause at each end (default 1000)\n"
        << "  --rate N          control rate Hz (default 100)\n"
        << "  --hold-end        hold final pose until Ctrl-C\n";
}

bool read7(int& i, int argc, char** argv, JointArray& out) {
    if (i + kArmJointCount >= argc) return false;
    for (int j = 0; j < kArmJointCount; ++j) out[j] = std::stof(argv[++i]);
    return true;
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const bool v = i + 1 < argc;
        if (s == "-h" || s == "--help") { usage(argv[0]); return false; }
        else if (s == "-r" || s == "--right") o.hand = 1;
        else if (s == "-l" || s == "--left") o.hand = 2;
        else if (s == "--port" && v) o.port = argv[++i];
        else if (s == "--baud" && v) o.baud = std::stoi(argv[++i]);
        else if (s == "--a" && v && std::strcmp(argv[i + 1], "cur") == 0) { ++i; o.a_current = true; o.a_set = true; }
        else if (s == "--a") { if (!read7(i, argc, argv, o.a)) return false; o.a_set = true; }
        else if (s == "--b") { if (!read7(i, argc, argv, o.b)) return false; o.b_set = true; o.b_rel = false; }
        else if (s == "--b-rel") { if (!read7(i, argc, argv, o.b)) return false; o.b_set = true; o.b_rel = true; }
        else if (s == "--vmax" && v) o.vmax = std::stof(argv[++i]);
        else if (s == "--kp" && v) o.kp = std::stof(argv[++i]);
        else if (s == "--kd" && v) o.kd = std::stof(argv[++i]);
        else if (s == "--max-err" && v) o.max_err = std::stof(argv[++i]);
        else if (s == "--margin" && v) o.margin = std::stof(argv[++i]);
        else if (s == "--cycles" && v) o.cycles = std::stoi(argv[++i]);
        else if (s == "--dwell-ms" && v) o.dwell_ms = std::stoi(argv[++i]);
        else if (s == "--rate" && v) o.rate_hz = std::stoi(argv[++i]);
        else if (s == "--hold-end") o.hold_end = true;
        else { std::cerr << "Unknown/incomplete option: " << s << "\n"; usage(argv[0]); return false; }
    }
    if (o.hand == 0) { std::cerr << "Need -r or -l\n"; return false; }
    if (!o.a_set || !o.b_set) { std::cerr << "Need --a and --b/--b-rel\n"; return false; }
    o.vmax = std::clamp(o.vmax, 0.01f, 1.0f);
    o.rate_hz = std::clamp(o.rate_hz, 20, 500);
    o.cycles = std::max(1, o.cycles);
    return true;
}

void printPose(const char* tag, const JointArray& q) {
    std::cout << std::fixed << std::setprecision(3) << tag;
    for (float x : q) std::cout << " " << x;
    std::cout << "\n";
}

bool withinLimits(const char* tag, const JointArray& q, float margin) {
    bool ok = true;
    for (int i = 0; i < kArmJointCount; ++i) {
        if (q[i] < kLower[i] + margin || q[i] > kUpper[i] - margin) {
            std::cerr << tag << ": j" << i << "=" << q[i] << " outside ["
                      << kLower[i] + margin << ", " << kUpper[i] - margin << "]\n";
            ok = false;
        }
    }
    return ok;
}

void sendCmd(ArmMotorBus& bus, const JointArray& q, const JointArray& dq, float kp, float kd) {
    std::array<ArmJointCmd, kArmJointCount> c{};
    for (int i = 0; i < kArmJointCount; ++i) {
        c[i].mode = 1; c[i].timed_out = false;
        c[i].q = q[i]; c[i].dq = dq[i]; c[i].tau = 0.f;
        c[i].kp = kp; c[i].kd = kd;
    }
    bus.step(c);
}

void sendStop(ArmMotorBus& bus) {
    std::array<ArmJointCmd, kArmJointCount> c{};
    for (auto& x : c) { x.mode = 0; x.timed_out = false; }
    bus.step(c);
}

JointArray measured(const ArmMotorBus& bus) {
    JointArray q{};
    const auto s = bus.states();
    for (int i = 0; i < kArmJointCount; ++i) q[i] = s[i].q;
    return q;
}

bool faulted(const ArmMotorBus& bus) {
    const auto s = bus.states();
    for (int i = 0; i < kArmJointCount; ++i) {
        if (s[i].mode != 1 || s[i].error_code != 0) {
            std::cerr << "Fault: j" << i << " mode=" << int(s[i].mode)
                      << " err=0x" << std::hex << s[i].error_code << std::dec
                      << " (err 0x8 = undervoltage: check PSU / current limit)\n";
            return true;
        }
    }
    return false;
}

// Quintic min-jerk: s(u)=10u^3-15u^4+6u^5, peak ds/du = 1.875 at u=0.5.
// Returns 0 ok, 1 fault/tracking, 2 interrupted.
int segment(ArmMotorBus& bus, const Options& o, const JointArray& from, const JointArray& to,
            const char* label) {
    float dmax = 0.f;
    for (int i = 0; i < kArmJointCount; ++i) dmax = std::max(dmax, std::fabs(to[i] - from[i]));
    const float T = std::max(1.0f, 1.875f * dmax / o.vmax);
    std::cout << label << ": max|dq|=" << std::setprecision(3) << dmax << " rad, duration "
              << std::setprecision(1) << T << " s (vmax " << std::setprecision(2) << o.vmax
              << " rad/s)\n";
    const auto period = std::chrono::microseconds(1000000 / o.rate_hz);
    const auto t0 = clock_t_::now();
    auto next = t0, last_print = t0;
    while (g_running) {
        const float t = std::chrono::duration<float>(clock_t_::now() - t0).count();
        const float u = std::min(1.f, t / T);
        const float s = u * u * u * (10.f - 15.f * u + 6.f * u * u);
        const float ds = 30.f * u * u * (1.f - u) * (1.f - u) / T;
        JointArray q{}, dq{};
        for (int i = 0; i < kArmJointCount; ++i) {
            q[i] = from[i] + (to[i] - from[i]) * s;
            dq[i] = (to[i] - from[i]) * ds;
        }
        sendCmd(bus, q, dq, o.kp, o.kd);
        if (faulted(bus)) return 1;
        const JointArray m = measured(bus);
        for (int i = 0; i < kArmJointCount; ++i) {
            if (std::fabs(q[i] - m[i]) > o.max_err) {
                std::cerr << "Abort: j" << i << " tracking error " << std::fabs(q[i] - m[i])
                          << " rad > " << o.max_err << " (raise --kp or lower --vmax)\n";
                return 1;
            }
        }
        if (clock_t_::now() - last_print >= std::chrono::seconds(1)) {
            std::cout << "  " << std::setprecision(0) << std::setw(3) << u * 100.f << "%";
            printPose("  q:", m);
            last_print = clock_t_::now();
        }
        if (u >= 1.f) return 0;
        next += period;
        std::this_thread::sleep_until(next);
    }
    return 2;
}

int dwell(ArmMotorBus& bus, const Options& o, const JointArray& q, int ms) {
    const JointArray zero{};
    const auto period = std::chrono::microseconds(1000000 / o.rate_hz);
    const auto t0 = clock_t_::now();
    auto next = t0;
    while (g_running && (ms < 0 || clock_t_::now() - t0 < std::chrono::milliseconds(ms))) {
        sendCmd(bus, q, zero, o.kp, o.kd);
        if (faulted(bus)) return 1;
        next += period;
        std::this_thread::sleep_until(next);
    }
    return g_running ? 0 : 2;
}

// Hold measured pose 0.5 s, then damping-only 1.5 s (gravity sinks slowly), then stop.
void gentleRelease(ArmMotorBus& bus, const Options& o) {
    std::cout << "Release: hold, then damping-only, then stop\n";
    const JointArray zero{};
    const JointArray q = measured(bus);
    const auto period = std::chrono::microseconds(1000000 / o.rate_hz);
    auto run = [&](int ms, float kp, float kd) {
        const auto t0 = clock_t_::now();
        auto next = t0;
        while (clock_t_::now() - t0 < std::chrono::milliseconds(ms)) {
            sendCmd(bus, q, zero, kp, kd);
            next += period;
            std::this_thread::sleep_until(next);
        }
    };
    run(500, o.kp, o.kd);
    run(1500, 0.f, std::max(o.kd, 2.f));
    for (int k = 0; k < 50; ++k) {
        sendStop(bus);
        std::this_thread::sleep_for(period);
    }
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        if (!parse(argc, argv, o)) return 1;
    } catch (const std::exception& e) {
        std::cerr << "Bad argument: " << e.what() << "\n";
        return 1;
    }
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    ArmMotorBus bus;
    if (!bus.open(o.port, o.baud, o.hand == 2 ? kLeftJointCalibPath : kRightJointCalibPath)) return 1;

    // Wait for feedback from all joints (read-only: send mode-0 stop).
    const auto t_fb = clock_t_::now();
    while (!bus.allFeedbackReady() && clock_t_::now() - t_fb < std::chrono::seconds(2)) {
        sendStop(bus);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!bus.allFeedbackReady()) {
        const auto r = bus.feedbackReady();
        std::cerr << "No feedback from joint(s):";
        for (int i = 0; i < kArmJointCount; ++i) if (!r[i]) std::cerr << " j" << i;
        std::cerr << "\nCheck port, A/B wiring, power. Nothing was moved.\n";
        bus.close();
        return 1;
    }

    const JointArray start = measured(bus);
    JointArray A = o.a_current ? start : o.a;
    JointArray B = o.b;
    if (o.b_rel) for (int i = 0; i < kArmJointCount; ++i) B[i] = A[i] + o.b[i];

    printPose("Current:", start);
    printPose("A      :", A);
    printPose("B      :", B);
    bool ok = withinLimits("A", A, o.margin) & withinLimits("B", B, o.margin);
    if (!o.a_current && !withinLimits("Current", start, 0.f)) {
        std::cerr << "(current pose is outside URDF limits: calibration file probably does not "
                     "match this arm)\n";
        ok = false;
    }
    if (!ok) {
        std::cerr << "Refusing to move. Nothing was moved.\n";
        bus.close();
        return 1;
    }

    // Enable FOC at the measured pose (no jump).
    {
        const JointArray zero{};
        const auto t0 = clock_t_::now();
        bool all = false;
        while (g_running && clock_t_::now() - t0 < std::chrono::seconds(1)) {
            sendCmd(bus, start, zero, o.kp, o.kd);
            const auto s = bus.states();
            all = std::all_of(s.begin(), s.end(), [](const ArmJointState& x) { return x.mode == 1; });
            if (all) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!all) {
            std::cerr << "Not all joints entered FOC. Try test_arm_lowlevel --clear / --reset.\n";
            for (int k = 0; k < 50; ++k) { sendStop(bus); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
            bus.close();
            return 1;
        }
    }
    std::cout << "FOC enabled at current pose.\n";

    int rc = 0;
    if (!o.a_current) rc = segment(bus, o, start, A, "Current -> A");
    if (rc == 0) rc = dwell(bus, o, A, o.dwell_ms);
    JointArray last = A;
    for (int c = 0; rc == 0 && c < o.cycles; ++c) {
        std::cout << "Cycle " << c + 1 << "/" << o.cycles << "\n";
        rc = segment(bus, o, A, B, "  A -> B");
        if (rc == 0) { last = B; rc = dwell(bus, o, B, o.dwell_ms); }
        if (rc == 0) rc = segment(bus, o, B, A, "  B -> A");
        if (rc == 0) { last = A; rc = dwell(bus, o, A, o.dwell_ms); }
    }
    if (rc == 0 && o.hold_end) {
        std::cout << "Holding final pose. Ctrl-C to release.\n";
        rc = dwell(bus, o, last, -1);
    }

    gentleRelease(bus, o);
    printPose("Final  :", measured(bus));
    bus.close();
    std::cout << (rc == 0 || rc == 2 ? "Done.\n" : "Aborted.\n");
    return rc == 1 ? 1 : 0;
}
