// Generate e+ e- -> mu+ mu- events with Pythia8 -- compiled and run INSIDE the
// hepstore/rivet-pythia container (which ships Pythia8 8.3 as a C++ library, no
// Python bindings). This is the *generation* stage of the Phase 2 orchestrated
// pipeline (clause b): a real, established generator, not the from-scratch toy.
//
// Writes one line per event -- the production cos(theta) of the mu- relative to
// the e- beam (+z) -- to EEMUMU_OUT, prefixed by a header carrying Pythia's own
// Monte-Carlo cross-section and the run parameters. The host-side analyze.py
// reads this and renders the labelled distribution.
//
// Process: the 2->2 s-channel WeakSingleBoson:ffbar2ffbar(s:gmZ), i.e.
// e+e- -> gamma*/Z -> f fbar. (The 2->1 resonance ffbar2gmZ underflows to zero at
// a fixed sqrt(s) far below the Z, because the Breit-Wigner integrates over a
// delta-function mHat.) The process sums all outgoing fermion flavours, so we
// select the mu+ mu- subset by the *primary* pair -- Pythia hard-process status
// code 23 -- which avoids counting muons that come from tau decays.
//
// Physics note (why this does NOT reproduce the toy's number): gamma*/Z carries a
// small Z-interference forward-backward asymmetry the pure-QED toy lacks, and the
// reported cross-section is summed over all final flavours. We switch the lepton
// beam PDF off so the collision sits at a fixed sqrt(s) (no ISR / energy spread),
// giving a clean teaching plot. The cross-check is qualitative (angular shape vs
// 1 + cos^2 theta), never a numerical cross-section equality.
//
// Build: g++ generate_pythia.cc -o gen $(pythia8-config --cxxflags --libs)

#include "Pythia8/Pythia.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

using namespace Pythia8;

static double env_d(const char* k, double dflt) {
  const char* v = std::getenv(k);
  return v ? std::atof(v) : dflt;
}
static int env_i(const char* k, int dflt) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : dflt;
}
static const char* env_s(const char* k, const char* dflt) {
  const char* v = std::getenv(k);
  return v ? v : dflt;
}

int main() {
  const double sqrtS = env_d("EEMUMU_SQRT_S", 10.0);
  const int nEvents = env_i("EEMUMU_N", 20000);
  const char* outPath = env_s("EEMUMU_OUT", "/tmp/eemumu_costheta.dat");
  const int seed = env_i("EEMUMU_SEED", 20260710);

  Pythia pythia;
  pythia.readString("Beams:idA = 11");   // e-
  pythia.readString("Beams:idB = -11");  // e+
  pythia.settings.parm("Beams:eCM", sqrtS);
  pythia.readString("WeakSingleBoson:ffbar2ffbar(s:gmZ) = on");  // 2->2 s-channel gamma*/Z
  pythia.readString("PDF:lepton = off");                         // fixed sqrt(s), no ISR
  pythia.readString("Random:setSeed = on");
  pythia.settings.mode("Random:seed", seed);
  pythia.readString("Print:quiet = on");
  pythia.readString("Next:numberCount = 0");

  if (!pythia.init()) {
    std::cerr << "Pythia failed to initialise\n";
    return 1;
  }

  std::vector<double> cosv;
  cosv.reserve(nEvents);
  for (int iE = 0; iE < nEvents; ++iE) {
    if (!pythia.next()) continue;
    // Primary hard-process mu- : status code 23 (outgoing hard particle), id 13.
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (p.id() == 13 && p.statusAbs() == 23) {
        const double pmag = std::sqrt(p.px() * p.px() + p.py() * p.py() + p.pz() * p.pz());
        if (pmag > 0.0) cosv.push_back(p.pz() / pmag);
        break;
      }
    }
  }

  const double sigma_mb = pythia.info.sigmaGen();   // mb
  const double sigma_err_mb = pythia.info.sigmaErr();
  const double version = pythia.parm("Pythia:versionNumber");

  // sigma_mb is Pythia's total ffbar cross-section (all flavours); n_events is the
  // mu+ mu- subset actually histogrammed. Both are recorded for honesty.
  std::ofstream fh(outPath);
  fh << "# process=ee->gmZ->ffbar(mumu_subset) sqrt_s_GeV=" << sqrtS
     << " n_events=" << cosv.size() << " sigma_mb=" << sigma_mb
     << " sigma_err_mb=" << sigma_err_mb << " pythia_version=" << version << "\n";
  for (double c : cosv) fh << c << "\n";

  std::cerr << "wrote " << cosv.size() << " events to " << outPath << "; Pythia sigma = "
            << sigma_mb * 1e6 << " +/- " << sigma_err_mb * 1e6 << " nb\n";
  return 0;
}
