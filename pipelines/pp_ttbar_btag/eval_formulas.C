// Evaluate card efficiency formulas with DELPHES' OWN evaluator (DelphesFormula,
// a TFormula subclass) on a (pt, eta) grid, so the host-side Python evaluator can
// be diffed against the authority rather than against a second guess.
//
//   ROOT_INCLUDE_PATH=/usr/local/venv/include \
//     root -l -b -q 'eval_formulas.C("formulas.txt","grid.txt","values.txt")'
//
// formulas.txt: records separated by a line "===FORMULA===". Backslash-newline
// continuations are stripped here (TCL does that before Delphes ever sees the
// string), so the Python side can send the formula exactly as parsed from the card.

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesFormula.h"
#endif

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void eval_formulas(const char* formulaFile, const char* gridFile, const char* outFile) {
  gSystem->Load("libDelphes");

  // --- read formula records ------------------------------------------------
  std::vector<std::string> formulas;
  {
    std::ifstream fh(formulaFile);
    std::string line, cur;
    while (std::getline(fh, line)) {
      // Files are written from Windows, so lines can carry a trailing CR.
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (line == "===FORMULA===") {
        if (!cur.empty()) formulas.push_back(cur);
        cur.clear();
      } else {
        cur += line + " ";
      }
    }
    if (!cur.empty()) formulas.push_back(cur);
  }

  // Strip TCL line-continuation backslashes (TCL removes them before Delphes).
  for (auto& f : formulas) {
    std::string clean;
    for (char c : f) clean += (c == '\\' || c == '\n' || c == '\t') ? ' ' : c;
    f = clean;
  }

  // --- read the (pt, eta) grid --------------------------------------------
  std::vector<std::pair<double, double>> grid;
  {
    std::ifstream fh(gridFile);
    double pt, eta;
    while (fh >> pt >> eta) grid.emplace_back(pt, eta);
  }

  std::ofstream out(outFile);
  out << "# DelphesFormula reference values: one line per (formula_index, pt, eta)\n";
  for (size_t i = 0; i < formulas.size(); ++i) {
    // NB: the (name, expression) constructor does NOT leave the formula ready to
    // execute -- Eval() then errors with "Formula is invalid". Delphes' own
    // modules default-construct and call Compile(), so do the same.
    DelphesFormula f;
    if (f.Compile(formulas[i].c_str()) != 0) {
      std::cerr << "FAILED to compile formula " << i << "\n";
      continue;
    }
    for (const auto& g : grid) {
      out << i << " " << std::setprecision(12) << g.first << " " << g.second << " "
          << f.Eval(g.first, g.second) << "\n";
    }
  }
  std::cerr << "evaluated " << formulas.size() << " formulas on " << grid.size()
            << " grid points\n";
}
