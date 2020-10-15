#include "merl_materials.h"

#include <algorithm>
#include <iostream>
#include <random>

namespace ibrdf {
// const std::vector<std::string> stones{ "aventurnine",
//                                        "black-obsidian",
//                                        "pink-jasper",
//                                        "white-marble" };

// const std::vector<std::string> metals{ "steel",
//                                        "chrome-steel",
//                                        "grease-covered-steel",
//                                        "chrome",
//                                        "aluminium",
//                                        "hematite",
//                                        "two-layer-gold",
//                                        "two-layer-silver",
//                                        "ss440",
//                                        "tungsten-carbide",
//                                        "brass",
//                                        "silicon-nitrade",
//                                        "nickel",
//                                        "alum-bronze",
//                                        "alumina-oxide",
//                                        "black-oxidized-steel" };

// const std::vector<std::string> paintings{ "red-matallic-paint",
//                                           "green-matallic-paint2",
//                                           "blue-metallic-paint2",
//                                           "color-changing-paint1",
//                                           "gold-metallic-paint3",
//                                           "color-changing-paint3",
//                                           "pearl-paint",
//                                           "yellow-paint",
//                                           "silver-paint",
//                                           "gold-paint",
//                                           "orange-paint",
//                                           "light-red-paint",
//                                           "dark-red-paint",
//                                           "dark-blue-paint",
//                                           "green-metallic-paint",
//                                           "purple-paint",
//                                           "white-paint",
//                                           "blue-metallic-paint2",
//                                           "silver-metallic-paint2",
//                                           "silver-metallic-paint",
//                                           "color-changing-paint2",
//                                           "gold-metallic-paint",
//                                           "gold-metallic-paint2",
//                                           "natural-209" };

// const std::vector<std::string> phenolics{
//   "specular-black-phenolic",   "black-phenolic",
//   "specular-blue-phenolic",    "specular-white-phenolic",
//   "specular-green-phenolic",   "red-phenolic",
//   "specular-organce-phenolic", "specular-maroon-phenolic",
//   "specular-red-phenolic",     "specular-violet-phenolic",
//   "specular-yellow-phenolic",  "yellow-phenolic"
// };

// const std::vector<std::string> acrylics{ "blue-acrylic",
//                                          "green-acrylic",
//                                          "violet-acrylic",
//                                          "white-acrylic" };

// const std::vector<std::string> plastics{ "yellow-matte-plastic",
//                                          "green-plastic",
//                                          "maroon-plastic",
//                                          "black-soft-plalstic",
//                                          "gray-plastic",
//                                          "red-plastic",
//                                          "yellow-plastic",
//                                          "pink-plastic",
//                                          "red-specular-plastic",
//                                          "pvc",
//                                          "delrin",
//                                          "polyethylene",
//                                          "polyurenthane-foam" };

// const std::vector<std::string> fabrics{ "dark-specular-fabric",
//                                         "light-brown-fabric",
//                                         "red-fabric",
//                                         "red-fabric2",
//                                         "green-fabric",
//                                         "black-fabric",
//                                         "blue-fabric",
//                                         "pink-fabric2",
//                                         "white-fabric2",
//                                         "beige-fabric",
//                                         "white-fabric2",
//                                         "pink-fabric",
//                                         "pink-felt",
//                                         "nylon" };

// const std::vector<std::string> rubbers{ "blue-rubber",        "pure-rubber",
//                                         "neoprene-rubber", "violet-rubber",
//                                         "green-latex",        "teflon",
//                                         "white-diffuse-bball" };

// const std::vector<std::string> woods{
//   "ipswich-pine-221", "special-walnut-224", "cherry-235",
//   "fruitwood-241",    "colonial-maple-223", "pickled-oak-260"
// };

// const std::vector<std::string> merlMaterials{
//   // Stones
//   "aventurnine",
//   "black-obsidian",
//   "pink-jasper",
//   "white-marble",

//   // Metals
//   "steel",
//   // "chrome-steel",
//   "grease-covered-steel",
//   "chrome",
//   // "aluminium",
//   "hematite",
//   "two-layer-gold",
//   "two-layer-silver",
//   // "ss440",
//   "tungsten-carbide",
//   // "brass",
//   "silicon-nitrade",
//   "nickel",
//   "alum-bronze",
//   "alumina-oxide",
//   "black-oxidized-steel",

//   // Paintings"
//   "red-metallic-paint",
//   // "green-matallic-paint2",
//   "blue-metallic-paint2",
//   // "color-changing-paint1",
//   "gold-metallic-paint3",
//   "color-changing-paint3",
//   "pearl-paint",
//   "yellow-paint",
//   "silver-paint",
//   "gold-paint",
//   "orange-paint",
//   "light-red-paint",
//   "dark-red-paint",
//   "dark-blue-paint",
//   "green-metallic-paint",
//   // "purple-paint",
//   // "white-paint",
//   "blue-metallic-paint2",
//   "silver-metallic-paint2",
//   // "silver-metallic-paint",
//   // "color-changing-paint2",
//   // "gold-metallic-paint",
//   "gold-metallic-paint2",
//   "natural-209",

//   // Phenolics
//   "specular-black-phenolic",
//   "black-phenolic",
//   "specular-blue-phenolic",
//   "specular-white-phenolic",
//   // "specular-green-phenolic",
//   "red-phenolic",
//   "specular-orange-phenolic",
//   "specular-maroon-phenolic",
//   "specular-red-phenolic",
//   "specular-violet-phenolic",
//   // "specular-yellow-phenolic",
//   "yellow-phenolic",

//   // Acrylics
//   "blue-acrylic",
//   "green-acrylic",
//   "violet-acrylic",
//   "white-acrylic",

//   // Plastics
//   "yellow-matte-plastic",
//   "green-plastic",
//   "maroon-plastic",
//   "black-soft-plastic",
//   // "gray-plastic",
//   "red-plastic",
//   "yellow-plastic",
//   // "pink-plastic",
//   "red-specular-plastic",
//   "pvc",
//   "delrin",
//   "polyethylene",
//   "polyurethane-foam",

//   // Fabrics
//   "dark-specular-fabric",
//   "light-brown-fabric",
//   // "red-fabric",
//   "red-fabric2",
//   "green-fabric",
//   "black-fabric",
//   "blue-fabric",
//   "pink-fabric2",
//   "white-fabric2",
//   "beige-fabric",
//   // "white-fabric",
//   // "pink-fabric",
//   "pink-felt",
//   "nylon",

//   // Rubbers
//   "blue-rubber",
//   "pure-rubber",
//   "neoprene-rubber",
//   "violet-rubber",
//   // "green-latex",
//   "teflon",
//   "white-diffuse-bball",

//   // Woods
//   "ipswich-pine-221",
//   "special-walnut-224",
//   "cherry-235",
//   "fruitwood-241",
//   "colonial-maple-223",
//   // "pickled-oak-260"
// };

std::vector<std::string>
SampleMaterials(const std::string& exclude, std::size_t numKeep)
{
  std::map<std::string, std::vector<std::string>> merlMaterials{
    { "stones",
      { "aventurnine", "black-obsidian", "pink-jasper", "white-marble" } },

    { "metals",
      { "steel",
        "chrome-steel",
        "grease-covered-steel",
        "chrome",
        "aluminium",
        "hematite",
        "two-layer-gold",
        "two-layer-silver",
        "ss440",
        "tungsten-carbide",
        "brass",
        "silicon-nitrade",
        "nickel",
        "alum-bronze",
        "alumina-oxide",
        "black-oxidized-steel" } },

    { "paintings",
      { "red-metallic-paint",
        "green-metallic-paint2",
        "blue-metallic-paint2",
        "color-changing-paint1",
        "gold-metallic-paint3",
        "color-changing-paint3",
        "pearl-paint",
        "yellow-paint",
        "silver-paint",
        "gold-paint",
        "orange-paint",
        "light-red-paint",
        "dark-red-paint",
        "dark-blue-paint",
        "green-metallic-paint",
        "purple-paint",
        "white-paint",
        "blue-metallic-paint2",
        "silver-metallic-paint2",
        "silver-metallic-paint",
        "color-changing-paint2",
        "gold-metallic-paint",
        "gold-metallic-paint2",
        "natural-209" } },

    { "phenolics",
      { "specular-black-phenolic",
        "black-phenolic",
        "specular-blue-phenolic",
        "specular-white-phenolic",
        "specular-green-phenolic",
        "red-phenolic",
        "specular-orange-phenolic",
        "specular-maroon-phenolic",
        "specular-red-phenolic",
        "specular-violet-phenolic",
        "specular-yellow-phenolic",
        "yellow-phenolic" } },

    { "acrylics",
      { "blue-acrylic", "green-acrylic", "violet-acrylic", "white-acrylic" } },

    { "plastics",
      { "yellow-matte-plastic",
        "green-plastic",
        "maroon-plastic",
        "black-soft-plastic",
        "gray-plastic",
        "red-plastic",
        "yellow-plastic",
        "pink-plastic",
        "red-specular-plastic",
        "pvc",
        "delrin",
        "polyethylene",
        "polyurethane-foam" } },

    { "fabrics",
      { "dark-specular-fabric",
        "light-brown-fabric",
        "red-fabric",
        "red-fabric2",
        "green-fabric",
        "black-fabric",
        "blue-fabric",
        "pink-fabric2",
        "white-fabric2",
        "beige-fabric",
        "white-fabric",
        "pink-fabric",
        "pink-felt",
        "nylon" } },

    { "rubbers",
      { "blue-rubber",
        "pure-rubber",
        "neoprene-rubber",
        "violet-rubber",
        "green-latex",
        "teflon",
        "white-diffuse-bball" } },

    { "woods",
      { "ipswich-pine-221",
        "special-walnut-224",
        "cherry-235",
        "fruitwood-241",
        "colonial-maple-223",
        "pickled-oak-260" } }
  };

  bool found = false;
  std::string category;

  for (auto const& [key, value] : merlMaterials) {
    for (auto const& m : value) {
      if (m == exclude) {
        category = key;
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }

  std::random_device rd;
  std::mt19937 g(rd());

  std::vector<std::string> ret;
  std::size_t size = 0;

  for (auto& [key, value] : merlMaterials) {
    if (key == category) {
      for (auto const& m : value) {
        if (m != exclude) {
          ret.emplace_back(m);
          ++size;
        }
      }
      continue;
    }

    std::shuffle(value.begin(), value.end(), g);
    std::size_t classSize = value.size();
    std::size_t toInsertSize =
      static_cast<std::size_t>(std::round(classSize * numKeep / 100.0f));
    if (size + toInsertSize > numKeep) {
      toInsertSize = numKeep - size;
    }

    ret.insert(ret.end(), value.begin(), value.begin() + toInsertSize);
    size += toInsertSize;
  }

  return ret;
}
}
