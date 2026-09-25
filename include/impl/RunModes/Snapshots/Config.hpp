#pragma once

// Include internal dependencies
#include "impl/Initialization/Init.hpp"

// Include external dependencies
#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Condor::impl::Modes::Snapshots {
    // Usings
    using string = std::string;
    using json = nlohmann::json;

    enum class Tracking_Mode {
        None,
        Interface
    };

    template <typename FloatType>
    struct Snapshots_Config {
        struct Settings {
            std::vector<FloatType> times;
            std::vector<std::pair<FloatType, int>> time_iterations;
            FloatType fallback_timestep = static_cast<FloatType>(1.0);
            Tracking_Mode tracking = Tracking_Mode::None;
        } settings;
        struct Output {
            bool temperature = false;
        } output;
        struct Hooks {
            json meltpool;
        } hooks;

        Snapshots_Config() = default;

        explicit Snapshots_Config(const json& mode_json, const FloatType longest_scan_time) {
            // Get the root of the Snapshots mode to initialize config on
            const json& root = Init::GetNestedJson(mode_json, "Snapshots", true);

            // Get different sections
            const json& settingsNode = Init::GetNestedJson(root, "settings", true);
            const json& outputNode = Init::GetNestedJson(root, "output", true);
            const json& hooksNode = Init::GetNestedJson(root, "hooks", false);

            // Read the ordered snapshot times before the timestep helpers are made.
            settings.times = ReadTimes(settingsNode, longest_scan_time);
            settings.time_iterations.reserve(settings.times.size());
            for (std::size_t i = 0; i < settings.times.size(); i++) {
                settings.time_iterations.emplace_back(settings.times[i], static_cast<int>(i) + 1);
            }
            settings.fallback_timestep = FallbackTimestep(settings.times, longest_scan_time);
            settings.tracking = ReadTrackingMode(settingsNode);

            // Read the shared output controls that Snapshots owns directly.
            output.temperature = Init::ReadValue<bool>(outputNode, "temperature", false, false);

            // Cache each hook node so later setup can defer any heavier parsing work.
            hooks.meltpool = Init::GetNestedJson(hooksNode, "meltpool", false);
        }

        private:
            static string Trim(const string& value) {
                const string::const_iterator begin = std::find_if_not(value.begin(), value.end(), [](const unsigned char c) {
                    return std::isspace(c);
                });
                const string::const_iterator end = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char c) {
                    return std::isspace(c);
                }).base();
                return (begin < end) ? string(begin, end) : string();
            }

            static FloatType ParseNumber(const string& text, const string& label) {
                std::size_t parsed = 0;
                const double value = std::stod(text, &parsed);
                if (parsed != text.size()) {
                    throw std::runtime_error("Snapshots time '" + label + "' contains extra characters.");
                }
                return static_cast<FloatType>(value);
            }

            static FloatType ReadTimeValue(const json& node, const FloatType longest_scan_time) {
                if (node.is_number()) {
                    return node.get<FloatType>();
                }
                if (!node.is_string()) {
                    throw std::runtime_error("Snapshots times must be numbers or percentage strings.");
                }

                const string original = node.get<string>();
                string text = Trim(original);
                const bool is_percentage = !text.empty() && text.back() == '%';
                if (is_percentage) {
                    text.pop_back();
                    text = Trim(text);
                }

                if (text.empty()) {
                    throw std::runtime_error("Snapshots time '" + original + "' is empty.");
                }

                const FloatType value = ParseNumber(text, original);
                return is_percentage
                    ? (value / static_cast<FloatType>(100.0)) * longest_scan_time
                    : value;
            }

            static std::vector<FloatType> ReadTimes(const json& settingsNode, const FloatType longest_scan_time) {
                if (!settingsNode.contains("times") || !settingsNode["times"].is_array()) {
                    throw std::runtime_error("Snapshots settings must contain a 'times' array.");
                }

                std::vector<FloatType> times;
                const json& timesNode = settingsNode["times"];
                times.reserve(timesNode.size());

                for (const json& timeNode : timesNode) {
                    const FloatType time = ReadTimeValue(timeNode, longest_scan_time);
                    if (!std::isfinite(static_cast<double>(time)) || time < static_cast<FloatType>(0.0)) {
                        throw std::runtime_error("Snapshots times must be finite and non-negative.");
                    }
                    if (!times.empty() && time <= times.back()) {
                        throw std::runtime_error("Snapshots times must be strictly increasing after percentage expansion.");
                    }
                    times.push_back(time);
                }

                if (times.empty()) {
                    throw std::runtime_error("Snapshots requires at least one snapshot time.");
                }
                return times;
            }

            static FloatType FallbackTimestep(const std::vector<FloatType>& times, const FloatType longest_scan_time) {
                if (!times.empty() && times.front() > static_cast<FloatType>(0.0)) {
                    return times.front();
                }
                for (std::size_t i = 1; i < times.size(); i++) {
                    const FloatType dt = times[i] - times[i - 1];
                    if (dt > static_cast<FloatType>(0.0)) {
                        return dt;
                    }
                }
                return (longest_scan_time > static_cast<FloatType>(0.0))
                    ? longest_scan_time
                    : static_cast<FloatType>(1.0);
            }

            static Tracking_Mode ReadTrackingMode(const json& settingsNode) {
                const string tracking = Init::ReadValue<string>(settingsNode, "tracking", "None", false);
                if (tracking == "None") {
                    return Tracking_Mode::None;
                }
                if (tracking == "Interface") {
                    return Tracking_Mode::Interface;
                }
                throw std::runtime_error("Snapshots tracking must be 'None' or 'Interface'.");
            }
    };
}
