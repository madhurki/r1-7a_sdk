// move_ab: slow, time-parameterised joint-space move A -> B (-> A ...) in ONE
// process, so FOC never drops between segments (no limp gaps, no sagging).
//
// Each segment is a quintic (minimum-jerk) profile: zero velocity and
// acceleration at both ends. Duration is chosen so the fastest joint never
// exceeds --vmax rad/s. Position + velocity feed-forward is sent every tick.
//
// Gravity compensation (--gcomp F): a feed-forward torque F x (model gravity torque) is added
// every tick, using link masses and centres of mass from Unitree's URDF. It removes the lag
// against gravity that otherwise makes the shoulder trail and then catch up.
//
// Ctrl-C (not a fault) drives the arm back to A along a smooth profile before releasing;
// a second Ctrl-C releases immediately.
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <poll.h>
#include <unistd.h>

namespace {

using JointArray = std::array<float, kArmJointCount>;
using clock_t_ = std::chrono::steady_clock;

// URDF joint limits (rad), same as demo_arm_motion.cpp.
constexpr JointArray kLower = {-2.723f, 0.f, -0.349f, 0.f, -2.269f, -0.785f, -1.571f};
constexpr JointArray kUpper = {2.723f, 3.1416f, 3.491f, 3.1415f, 2.269f, 0.785f, 1.571f};

// Kinematic and mass parameters from Unitree URDF V1.2, right arm (urdf/URDF_R1-7A.tar.gz).
constexpr float kJointXyz[7][3] = {{0.0f, 0.0f, 0.069f}, {0.0f, 0.0f, 0.0469f}, {0.0f, 0.1085f, -0.0047926f}, {0.025f, 0.082408f, 0.016191f}, {-0.022314f, -0.10786f, 0.062877f}, {-1.1462e-05f, -0.1072f, 0.00069991f}, {0.0f, -0.01f, 0.0f}};
constexpr float kJointRpy[7][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {-0.017453f, 0.0f, 0.0f}, {0.0f, -0.016375f, 0.0f}, {-3.1416f, 0.0f, -3.1416f}};
constexpr float kJointAxis[7][3] = {{0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}};
constexpr float kLinkMass[7] = {0.89452f, 0.74773f, 0.71845f, 0.85572f, 0.59294f, 0.391f, 0.1048f};  // Link1..Link7, kg
constexpr float kLinkCom[7][3] = {{2.156e-08f, -9.6233e-09f, 0.045314f}, {0.0017013f, 0.063321f, -0.0042531f}, {-0.0062069f, 0.075751f, 0.012556f}, {-0.020915f, -0.070455f, 0.060056f}, {0.00042498f, -0.043121f, 0.00020709f}, {-1.3033e-07f, -0.0098039f, 8.3267e-17f}, {-3.7143e-07f, -0.018577f, 0.00595f}};  // in each link frame, m

void rpyToR(const float* r, float* R) {  // R = Rz(y) Ry(p) Rx(r), row-major
    const float cr = std::cos(r[0]), sr = std::sin(r[0]), cp = std::cos(r[1]), sp = std::sin(r[1]),
                cy = std::cos(r[2]), sy = std::sin(r[2]);
    const float M[9] = {cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                        -sp, cp * sr, cp * cr};
    std::memcpy(R, M, sizeof(M));
}
void axisAngleToR(const float* a, float t, float* R) {
    const float c = std::cos(t), s = std::sin(t), C = 1.f - c, x = a[0], y = a[1], z = a[2];
    const float M[9] = {x * x * C + c, x * y * C - z * s, x * z * C + y * s,
                        y * x * C + z * s, y * y * C + c, y * z * C - x * s,
                        z * x * C - y * s, z * y * C + x * s, z * z * C + c};
    std::memcpy(R, M, sizeof(M));
}
void mul33(const float* A, const float* B, float* C) {
    float T[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) T[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    std::memcpy(C, T, sizeof(T));
}
void mulv(const float* R, const float* v, float* o) {
    for (int i = 0; i < 3; ++i) o[i] = R[i * 3] * v[0] + R[i * 3 + 1] * v[1] + R[i * 3 + 2] * v[2];
}

// Torque each joint must supply to hold the arm still against gravity at pose q (URDF sign, Nm).
JointArray gravityTorque(const JointArray& q) {
    float R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, p[3] = {0, 0, 0};
    float jp[7][3], jz[7][3], cw[7][3];
    for (int i = 0; i < kArmJointCount; ++i) {
        float Ro[9], Rj[9], d[3], Ra[9];
        rpyToR(kJointRpy[i], Ro);
        mulv(R, kJointXyz[i], d);
        for (int k = 0; k < 3; ++k) p[k] += d[k];
        mul33(R, Ro, Rj);
        mulv(Rj, kJointAxis[i], jz[i]);
        std::memcpy(jp[i], p, sizeof(p));
        axisAngleToR(kJointAxis[i], q[i], Ra);
        mul33(Rj, Ra, R);
        mulv(R, kLinkCom[i], d);
        for (int k = 0; k < 3; ++k) cw[i][k] = p[k] + d[k];
    }
    JointArray tau{};
    for (int k = 0; k < kArmJointCount; ++k) {
        float t = 0.f;
        for (int l = k; l < kArmJointCount; ++l) {
            const float r[3] = {cw[l][0] - jp[k][0], cw[l][1] - jp[k][1], cw[l][2] - jp[k][2]};
            const float f[3] = {0.f, 0.f, -9.81f * kLinkMass[l]};
            const float m[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
            t += m[0] * jz[k][0] + m[1] * jz[k][1] + m[2] * jz[k][2];
        }
        tau[k] = -t;  // the motor must cancel gravity's moment
    }
    return tau;
}

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
    bool one_way = false;    // go A -> B, hold at B until Ctrl-C, then return to A
    float gcomp = 0.f;       // gravity feed-forward scale (0 = off, 1 = full model)
    bool model_only = false; // print model gravity torques for --b and exit
    std::string waypoints_path;  // one pose per line (7 rad); visited in order from the start pose
    bool sync = false;           // after each waypoint print READY and wait for a line on stdin
    int settle_ms = 1500;        // hold before measuring at each waypoint
    std::vector<JointArray> waypoints;
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
        << "  --hold-end        hold final pose until Ctrl-C\n"
        << "  --one-way         go A -> B and hold there; Ctrl-C returns smoothly to A\n"
        << "  --gcomp F         gravity compensation scale, 0..1.2 (default 0 = off; try 1.0)\n"
        << "  --model q0..q6    print the model's gravity torques at q and exit (no arm needed)\n"
        << "  --waypoints FILE  visit each pose in FILE (7 rad per line, # comments), then return\n"
        << "  --sync            at each waypoint print 'READY k q0..q6' and wait for Enter on stdin\n"
        << "  --settle-ms N     hold time before measuring at each waypoint (default 1500)\n"
        << "Ctrl-C during a move or hold returns smoothly to A; press it again to release at once.\n";
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
        else if (s == "--one-way") o.one_way = true;
        else if (s == "--waypoints" && v) o.waypoints_path = argv[++i];
        else if (s == "--sync") o.sync = true;
        else if (s == "--settle-ms" && v) o.settle_ms = std::stoi(argv[++i]);
        else if (s == "--gcomp" && v) o.gcomp = std::stof(argv[++i]);
        else if (s == "--model") { if (!read7(i, argc, argv, o.b)) return false; o.model_only = true; }
        else { std::cerr << "Unknown/incomplete option: " << s << "\n"; usage(argv[0]); return false; }
    }
    if (o.model_only) return true;
    if (o.hand == 0) { std::cerr << "Need -r or -l\n"; return false; }
    if (!o.waypoints_path.empty()) {
        std::ifstream f(o.waypoints_path);
        if (!f) { std::cerr << "Cannot open " << o.waypoints_path << "\n"; return false; }
        std::string line;
        int ln = 0;
        while (std::getline(f, line)) {
            ++ln;
            const auto h = line.find('#');
            if (h != std::string::npos) line.resize(h);
            std::istringstream ss(line);
            JointArray q{};
            int n = 0;
            while (n < kArmJointCount && ss >> q[n]) ++n;
            if (n == 0) continue;
            if (n != kArmJointCount) { std::cerr << o.waypoints_path << ":" << ln << ": need 7 numbers\n"; return false; }
            o.waypoints.push_back(q);
        }
        if (o.waypoints.empty()) { std::cerr << "No poses in " << o.waypoints_path << "\n"; return false; }
        if (!o.a_set) { o.a_current = true; o.a_set = true; }
        o.b_set = true;
        o.b = o.waypoints.back();
        o.b_rel = false;
        o.one_way = false;
        o.settle_ms = std::max(200, o.settle_ms);
    }
    if (!o.a_set || !o.b_set) { std::cerr << "Need --a and --b/--b-rel\n"; return false; }
    o.vmax = std::clamp(o.vmax, 0.01f, 1.0f);
    o.rate_hz = std::clamp(o.rate_hz, 20, 500);
    o.cycles = std::max(1, o.cycles);
    o.gcomp = std::clamp(o.gcomp, 0.f, 1.2f);
    if (o.gcomp > 0.f && o.hand == 2)
        std::cerr << "Note: the gravity model uses the right-arm URDF; check it before trusting it on a left arm.\n";
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

void sendCmd(ArmMotorBus& bus, const JointArray& q, const JointArray& dq, float kp, float kd,
             const JointArray& tau = JointArray{}) {
    std::array<ArmJointCmd, kArmJointCount> c{};
    for (int i = 0; i < kArmJointCount; ++i) {
        c[i].mode = 1; c[i].timed_out = false;
        c[i].q = q[i]; c[i].dq = dq[i]; c[i].tau = tau[i];
        c[i].kp = kp; c[i].kd = kd;
    }
    bus.step(c);
}

void sendStop(ArmMotorBus& bus) {
    std::array<ArmJointCmd, kArmJointCount> c{};
    for (auto& x : c) { x.mode = 0; x.timed_out = false; }
    bus.step(c);
}

JointArray gff(const Options& o, const JointArray& q) {
    JointArray t{};
    if (o.gcomp <= 0.f) return t;
    const JointArray g = gravityTorque(q);
    for (int i = 0; i < kArmJointCount; ++i) t[i] = o.gcomp * g[i];
    return t;
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
        sendCmd(bus, q, dq, o.kp, o.kd, gff(o, q));
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
    auto next = t0, last_print = t0;
    const JointArray ff = gff(o, q), model = gravityTorque(q);
    while (g_running && (ms < 0 || clock_t_::now() - t0 < std::chrono::milliseconds(ms))) {
        sendCmd(bus, q, zero, o.kp, o.kd, ff);
        if (faulted(bus)) return 1;
        if (ms < 0 && clock_t_::now() - last_print >= std::chrono::seconds(1)) {
            // While holding still, measured joint torque should match the gravity model.
            const auto st = bus.states();
            std::cout << std::fixed << std::setprecision(2) << "  hold  tau meas/model  j1 " << st[1].tau_est << "/"
                      << model[1] << "  j2 " << st[2].tau_est << "/" << model[2] << "  j3 " << st[3].tau_est << "/"
                      << model[3] << " Nm\n";
            last_print = clock_t_::now();
        }
        next += period;
        std::this_thread::sleep_until(next);
    }
    return g_running ? 0 : 2;
}

// Hold measured pose 0.5 s, then damping-only 1.5 s (gravity sinks slowly), then stop.
// Hold q for n ticks and return the mean measured pose.
int holdAverage(ArmMotorBus& bus, const Options& o, const JointArray& q, int n, JointArray& avg) {
    const JointArray zero{}, ff = gff(o, q);
    const auto period = std::chrono::microseconds(1000000 / o.rate_hz);
    auto next = clock_t_::now();
    avg = JointArray{};
    for (int k = 0; k < n && g_running; ++k) {
        sendCmd(bus, q, zero, o.kp, o.kd, ff);
        if (faulted(bus)) return 1;
        const JointArray m = measured(bus);
        for (int i = 0; i < kArmJointCount; ++i) avg[i] += m[i] / n;
        next += period;
        std::this_thread::sleep_until(next);
    }
    return g_running ? 0 : 2;
}

// Hold q until a full line arrives on stdin. EOF counts as a request to stop (returns 2).
int holdUntilLine(ArmMotorBus& bus, const Options& o, const JointArray& q) {
    const JointArray zero{}, ff = gff(o, q);
    const auto period = std::chrono::microseconds(1000000 / o.rate_hz);
    auto next = clock_t_::now();
    while (g_running) {
        sendCmd(bus, q, zero, o.kp, o.kd, ff);
        if (faulted(bus)) return 1;
        pollfd pfd{STDIN_FILENO, POLLIN, 0};
        if (poll(&pfd, 1, 0) > 0) {
            char c;
            const ssize_t r = ::read(STDIN_FILENO, &c, 1);
            if (r <= 0) return 2;
            if (c == '\n') return 0;
            continue;  // keep draining the line without sleeping
        }
        next += period;
        std::this_thread::sleep_until(next);
    }
    return 2;
}

void gentleRelease(ArmMotorBus& bus, const Options& o) {
    std::cout << "Release: hold, then damping-only, then stop\n";
    const JointArray zero{};
    const JointArray q = measured(bus);
    const auto period = std::chrono::microseconds(1000000 / o.rate_hz);
    const JointArray ff = gff(o, q);
    auto run = [&](int ms, float kp, float kd, bool with_g) {
        const auto t0 = clock_t_::now();
        auto next = t0;
        while (clock_t_::now() - t0 < std::chrono::milliseconds(ms)) {
            sendCmd(bus, q, zero, kp, kd, with_g ? ff : zero);
            next += period;
            std::this_thread::sleep_until(next);
        }
    };
    run(500, o.kp, o.kd, true);
    run(1500, 0.f, std::max(o.kd, 2.f), false);
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
    if (o.model_only) {
        const JointArray g = gravityTorque(o.b);
        printPose("q            :", o.b);
        std::cout << std::fixed << std::setprecision(3) << "gravity (Nm) :";
        for (float x : g) std::cout << " " << x;
        std::cout << "\n";
        return 0;
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
    // Targets must stay --margin inside the limits. When A is the current pose it only has to be
    // inside the hard limits (with 0.05 rad slack for encoder noise): the folded rest pose sits
    // exactly on the j1/j3 lower limit, and moving away from it must be allowed.
    const bool a_ok = o.a_current ? withinLimits("A (current pose)", A, -0.05f)
                                  : withinLimits("A", A, o.margin);
    bool ok = a_ok & withinLimits("B", B, o.margin);
    for (size_t k = 0; k < o.waypoints.size(); ++k)
        ok &= withinLimits(("waypoint " + std::to_string(k)).c_str(), o.waypoints[k], o.margin);
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
            sendCmd(bus, start, zero, o.kp, o.kd, gff(o, start));
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
    if (o.sync) std::cout << std::unitbuf;  // a parent process reads our output line by line
    if (!o.a_current) rc = segment(bus, o, start, A, "Current -> A");
    if (rc == 0) rc = dwell(bus, o, A, o.dwell_ms);
    JointArray last = A;
    if (rc == 0 && !o.waypoints.empty()) {
        JointArray prev = A;
        for (size_t k = 0; rc == 0 && k < o.waypoints.size(); ++k) {
            const std::string label = "  -> waypoint " + std::to_string(k);
            rc = segment(bus, o, prev, o.waypoints[k], label.c_str());
            if (rc == 0) rc = dwell(bus, o, o.waypoints[k], o.settle_ms);
            JointArray avg{};
            if (rc == 0) rc = holdAverage(bus, o, o.waypoints[k], 30, avg);
            if (rc == 0) {
                std::cout << "READY " << k << std::fixed << std::setprecision(5);
                for (float x : avg) std::cout << " " << x;
                std::cout << std::endl;
                if (o.sync) rc = holdUntilLine(bus, o, o.waypoints[k]);
            }
            prev = o.waypoints[k];
        }
        if (rc == 0) {
            rc = segment(bus, o, prev, A, "  -> start");
            if (rc == 0) dwell(bus, o, A, 300);
        }
    }
    if (rc == 0 && o.one_way) {
        rc = segment(bus, o, A, B, "  A -> B");
        if (rc == 0) {
            last = B;
            std::cout << "Holding at B. Ctrl-C to return smoothly to A.\n";
            rc = dwell(bus, o, B, -1);
        }
    }
    for (int c = 0; o.waypoints.empty() && !o.one_way && rc == 0 && c < o.cycles; ++c) {
        std::cout << "Cycle " << c + 1 << "/" << o.cycles << "\n";
        rc = segment(bus, o, A, B, "  A -> B");
        if (rc == 0) { last = B; rc = dwell(bus, o, B, o.dwell_ms); }
        if (rc == 0) rc = segment(bus, o, B, A, "  B -> A");
        if (rc == 0) { last = A; rc = dwell(bus, o, A, o.dwell_ms); }
    }
    if (rc == 0 && o.hold_end && !o.one_way) {
        std::cout << "Holding final pose. Ctrl-C to release.\n";
        rc = dwell(bus, o, last, -1);
    }

    if (rc == 2) {  // Ctrl-C, not a fault: bring the arm back to A under control first
        g_running = true;
        std::cout << "\nCtrl-C: returning smoothly to A (press Ctrl-C again to release now)\n";
        const int r2 = segment(bus, o, measured(bus), A, "  -> A");
        if (r2 == 0) dwell(bus, o, A, 300);
        rc = (r2 == 1) ? 1 : 2;
    }
    (void)last;
    gentleRelease(bus, o);
    printPose("Final  :", measured(bus));
    bus.close();
    std::cout << (rc == 0 || rc == 2 ? "Done.\n" : "Aborted.\n");
    return rc == 1 ? 1 : 0;
}
