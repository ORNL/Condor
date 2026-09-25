#pragma once

// Include necessary structures
#include "impl/Structs/Simdat.hpp"
#include <string>
#include <iostream>
#include <fstream>

namespace Condor::impl{
	namespace Out {
		// Usings
		using std::string;

		// Class for handling logging and printing (based on level)
		class Logger {
		private:
			std::streambuf* original_buf;
			std::ofstream file_stream;
			bool is_muted = false;

		public:
			Logger(const int rank, const int print_level) {
				// Save the original console buffer
				original_buf = std::cout.rdbuf();

				if (print_level == 0) {
					// Level 0: Mute everyone
					std::cout.setstate(std::ios_base::failbit);
					is_muted = true;
				} 
				else if (print_level == 1) {
					// Level 1: Only rank 0 prints to console
					if (rank != 0) {
						std::cout.setstate(std::ios_base::failbit);
						is_muted = true;
					}
				} 
				else if (print_level == 2) {
					// Level 2: Every rank to its own file
					std::string filename = "rank_" + std::to_string(rank) + ".log";
					file_stream.open(filename);
					if (file_stream.is_open()) {
						std::cout.rdbuf(file_stream.rdbuf());
						std::cout << std::unitbuf; // Flush every line
					}
				}
			}

			// Destructor: Automatically called when the object goes out of scope
			~Logger() {
				if (is_muted) {
					std::cout.clear(); // Remove the failbit
				}
				// Restore original buffer
				std::cout.rdbuf(original_buf);
			
				if (file_stream.is_open()) {
					file_stream.close();
				}
			}
		};
	}
}
