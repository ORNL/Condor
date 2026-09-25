#pragma once

// External includes
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace Condor::impl{
    namespace Init{
		// Usings and namespaces
		using json = nlohmann::json;
		namespace fs = std::filesystem;
		using string = std::string;
		
		// Helper function for reading JSON files
		inline json ReadJsonFile(const string& configFile) {
			std::ifstream f(configFile);
			if (!f.is_open()) {
				throw std::runtime_error("Could not open input file: " + configFile);
			}

			// Parse the JSON file
			json root = json::parse(f);

			// Return the parsed JSON
			return root;
		}

		// Helper function to extracted nested JSON objects and throw exception if a critical object is missing. Non critical objects can be left empty.
		inline const json& GetNestedJson(const json& j, const std::string& key, bool isCritical) {
			// This static object provides a safe memory address to return 
			static const json empty_object = json::object();

			// Check for key and make sure it's an object
			if (j.contains(key) && j[key].is_object()) {
				return j[key];
			}

			// If we get here, return error if it's critical
			if (isCritical) {
				throw std::runtime_error("Critical Input Error: Missing required object block '" + key + "'");
			}

			// Return empty object if it's not critical
			return empty_object;
		}

		//Helper to extract values from JSON and throw exception if a critical field is missing. Non critical fields can be left empty or set to a default value.
		template <typename T>
		inline T ReadValue(const json& j, const string& key, const T& defaultValue, bool isCritical) {
			if (j.contains(key) && !j[key].is_null()) {
				return j[key].get<T>();
			}
			if (isCritical) {
				throw std::runtime_error("Critical Input Error: Missing required field '" + key + "'");
			}
			return defaultValue;
		}
    }
}