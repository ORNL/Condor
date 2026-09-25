#pragma once

// Include internal dependenices
#include "impl/Initialization/JsonUtility.hpp"

// Include external dependencies
#include <nlohmann/json.hpp>
#include <string>
#include <stdexcept>
#include <vector>
#include <mpi.h>
#include <limits>


namespace Condor::impl::Init {

    using json = nlohmann::json;
    using string = std::string;

    template<typename T>
    using vector = std::vector<T>;

    // Structure containing raw simulation inputs before Simdat initialization.
    class FileReader 
    {
    public:
        string name = "TestSim";
        string dataDir;

        json mode_json = json::object();
        vector<json> beam_jsons;
        vector<string> path_strings;

        json material_json = json::object();
        json domain_json = json::object();
        json settings_json = json::object();

        bool has_settings = false;

        void Initialize(const string& configFile) {
            // Find the data directory
            dataDir = (fs::path(configFile).parent_path() / "Data").string();

            // Read the main file
            const json root = ReadJsonFile(configFile);

            // Read in name
            name = ReadValue<string>(root, "name", string("TestSim"), false);

            // Read mode
            mode_json = readFileOrJson(root, "mode", true);

            // Read material
            material_json = readFileOrJson(root, "material", true);

            // Read domain
            domain_json = readFileOrJson(root, "domain", true);

            // Read settings (not critical)
            settings_json = readFileOrJson(root, "settings", false);
            has_settings = !settings_json.empty();

            // Read beams and paths
            readBeamsAndPaths(root, beam_jsons, path_strings);

            // Validate the inputs
            Validate();
        }

        void Validate() const {
            if (!mode_json.is_object() || mode_json.empty()) {
                throw std::runtime_error("FileReader validation error: mode_json must be a non-empty object.");
            }
            if (!material_json.is_object() || material_json.empty()) {
                throw std::runtime_error("FileReader validation error: material_json must be a non-empty object.");
            }
            if (!domain_json.is_object() || domain_json.empty()) {
                throw std::runtime_error("FileReader validation error: domain_json must be a non-empty object.");
            }
            if (beam_jsons.empty()) {
                throw std::runtime_error("FileReader validation error: beam_jsons must contain at least one beam.");
            }
            if (path_strings.empty()) {
                throw std::runtime_error("FileReader validation error: path_strings must contain at least one path.");
            }
            if (beam_jsons.size() != path_strings.size()) {
                throw std::runtime_error("FileReader validation error: beam_jsons and path_strings must have matching sizes.");
            }
        }

        // Make a temporary json for easy communication
        json Pack() const {
            return json{
                {"name", name},
                {"dataDir", dataDir},
                {"mode_json", mode_json},
                {"beam_jsons", beam_jsons},
                {"path_strings", path_strings},
                {"material_json", material_json},
                {"domain_json", domain_json},
                {"settings_json", settings_json},
                {"has_settings", has_settings}
            };
        }

        void Unpack(const json& packed) {
            name = packed.value("name", string("TestSim"));
            dataDir = packed.value("dataDir", string());

            mode_json = packed.at("mode_json");
            beam_jsons = packed.at("beam_jsons").get<vector<json>>();
            path_strings = packed.at("path_strings").get<vector<string>>();

            material_json = packed.at("material_json");
            domain_json = packed.at("domain_json");
            settings_json = packed.value("settings_json", json::object());

            has_settings = packed.value("has_settings", false);

            Validate();
        }

        void Broadcast(const int root, MPI_Comm comm)
        {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);

            string payload;
            int payloadSize = 0;

            if (rank == root) {
                payload = Pack().dump();
                if (payload.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("FileReader broadcast payload is too large for MPI_Bcast.");
                }
                payloadSize = static_cast<int>(payload.size());
            }

            // Broadcast the size and resize appropirately on non-root ranks
            MPI_Bcast(&payloadSize, 1, MPI_INT, root, comm);
            if (rank != root){
                payload.resize(static_cast<size_t>(payloadSize));
            }
            
            // Broadcast the actual data and unpack on non-root ranks
            MPI_Bcast(payload.data(), payloadSize, MPI_CHAR, root, comm);
            if (rank != root) {
                Unpack(json::parse(payload));
            }
        }

    private:
        // Function for reading a JSON input which can be either a filename or a direct JSON object.
        json readFileOrJson(const json& node, const string& key, const bool isCritical)
        {
            if (!node.contains(key) || node[key].is_null()) {
                if (isCritical) {
                    throw std::runtime_error("Input '" + key + "' is required.");
                }
                return json::object();
            }

            if (node[key].is_string()) {
                return ReadJsonFile(node[key].get<string>());
            }
            if (node[key].is_object()) {
                return node[key];
            }

            throw std::runtime_error("Input '" + key + "' must be a filename string or JSON object.");
        }

        // Function for reading multiple files with wildcard support.
        vector<string> resolveWildcards(const string& pattern)
        {
            vector<string> files;
            const size_t pos = pattern.find('*');

            if (pos == string::npos) {
                files.push_back(pattern);
                return files;
            }

            const string prefix = pattern.substr(0, pos);
            const string suffix = pattern.substr(pos + 1);
            if (suffix.find('*') != string::npos) {
                throw std::runtime_error("Input pattern contains multiple wildcards: " + pattern);
            }

            for (int num = 1;; ++num) {
                const string file = prefix + std::to_string(num) + suffix;
                if (!fs::exists(file)) break;
                files.push_back(file);
            }

            if (files.empty()) {
                throw std::runtime_error("No files found matching input pattern: " + pattern);
            }
            return files;
        }

        // Function for quickly reading a text file into a string.
        string readTextFile(const string& filePath)
        {
            std::ifstream file(filePath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + filePath);
            }

            const std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            string buffer;
            buffer.resize(static_cast<size_t>(size));

            if (size > 0 && !file.read(buffer.data(), size)) {
                throw std::runtime_error("Error occurred while reading: " + filePath);
            }

            return buffer;
        }

        // Function for reading beams and paths, which can be either filenames or direct JSON/text.
        void readBeamsAndPaths(const json& root, vector<json>& beams, vector<string>& paths)
        {
            beams.clear();
            paths.clear();

            if (!root.contains("beam") || root["beam"].is_null()) {
                throw std::runtime_error("Input 'beam' is required.");
            }
            if (!root.contains("path") || root["path"].is_null()) {
                throw std::runtime_error("Input 'path' is required.");
            }

            const auto readBeamNode = [&](const json& node) {
                if (node.is_string()) {
                    const vector<string> beamFiles = resolveWildcards(node.get<string>());
                    for (const string& file : beamFiles) {
                        beams.push_back(ReadJsonFile(file));
                    }
                    return;
                }
                if (node.is_object()) {
                    beams.push_back(node);
                    return;
                }
                throw std::runtime_error("Input 'beam' must be a filename string, JSON object, or array.");
            };

            const auto readPathNode = [&](const json& node) {
                if (node.is_string()) {
                    const vector<string> pathFiles = resolveWildcards(node.get<string>());
                    for (const string& file : pathFiles) {
                        paths.push_back(readTextFile(file));
                    }
                    return;
                }
                if (node.is_object()) {
                    paths.push_back(node.dump());
                    return;
                }
                throw std::runtime_error("Input 'path' must be a filename string, JSON object, or array.");
            };

            if (root["beam"].is_array()) {
                for (const json& beamNode : root["beam"]) {
                    readBeamNode(beamNode);
                }
            } else {
                readBeamNode(root["beam"]);
            }

            if (root["path"].is_array()) {
                for (const json& pathNode : root["path"]) {
                    readPathNode(pathNode);
                }
            } else {
                readPathNode(root["path"]);
            }

            if (beams.size() != paths.size()) {
                throw std::runtime_error(
                    "Number of beam inputs and path inputs must match. Found " +
                    std::to_string(beams.size()) + " beam inputs and " +
                    std::to_string(paths.size()) + " path inputs.");
            }
        }
    };
}
