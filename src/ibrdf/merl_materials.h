#pragma once

#include <map>
#include <string>
#include <vector>

namespace ibrdf {
// extern const std::vector<std::string> stones;
// extern const std::vector<std::string> metals;
// extern const std::vector<std::string> paintings;
// extern const std::vector<std::string> phenolics;
// extern const std::vector<std::string> acrylics;
// extern const std::vector<std::string> plastics;
// extern const std::vector<std::string> fabrics;
// extern const std::vector<std::string> rubbers;
// extern const std::vector<std::string> woods;

// 9 classes
// 4  0 1
// 16 4 7
// 24 7 13
// 12 2 4
// 4  0 1
// 13 2 5
// 14 3 5
// 7  1 2
// 6  1 2
// extern const std::map<std::string, std::vector<std::string>> merlMaterials;
std::vector<std::string>
SampleMaterials(const std::string& exclude, std::size_t numKeep);
}
