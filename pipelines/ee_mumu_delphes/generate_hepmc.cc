// Generate e+ e- -> mu+ mu- events with Pythia8 at sqrt(s) = 250 GeV and write a
// full-event HepMC3 file (+ a small meta header) -- compiled and run INSIDE the
// hepstore/rivet-pythia container (Pythia8 8.3 + HepMC3 as a C++ library). This is
// the generation stage of the Phase 2 *detector* extension: the HepMC3 file is fed
// to Delphes (in a second, established image) for fast detector simulation.
//
// The truth AND reco cos(theta) distributions are BOTH produced downstream by the
// Delphes ROOT macro (extract_reco.C), from Delphes' generator "Particle" branch
// and reco "Muon" branch of the *same* events -- so the two populations are
// identical up to detector response (see that macro for why: status 23 is not
// preserved through the HepMC round-trip, and the signal mu- is isolated by an
// angle-neutral |p| cut, not by status). Here we only emit:
//   EEMUMU_META   ("<...>.dat")   -- a one-line header: Pythia's MC cross-section
//                                    (all flavours) + run params + the generator's
//                                    own primary-mu- count (status 23) as an
//                                    independent cross-check of the macro's truth.
//   EEMUMU_HEPMC  ("<...>.hepmc")  -- HepMC3 ascii3 stream of every event.
//
// Energy: sqrt(s) = 250 GeV (ILC). This is a DELIBERATE change from the
// clause-(b) chain (10 GeV): the standard Delphes e+e- cards (ILD/IDEA/CLIC) are
// parametrized for >= 91 GeV, so a physically meaningful detector response needs
// an ILC-scale energy -- see pipelines/ee_mumu_delphes/README.md. At 250 GeV the
// process is well above the Z, so gamma-Z interference gives a *sizeable*
// forward-backward asymmetry (the mu- is forward-peaked): the angular spectrum is
// NOT the symmetric 1 + cos^2(theta) of the 10 GeV chain, and A_FB is measured,
// not assumed.
//
// Process: the 2->2 s-channel WeakSingleBoson:ffbar2ffbar(s:gmZ), i.e.
// e+e- -> gamma*/Z -> f fbar. We pick the mu+ mu- subset by the *primary* pair
// (Pythia hard-process status code 23), avoiding muons from tau decays. The
// reported cross-section is the all-flavour ffbar total (as in the 10 GeV chain),
// so no numerical sigma-equality is asserted; the deliverable is the truth-vs-reco
// angular comparison.
//
// Build: g++ generate_hepmc.cc -o gen $(pythia8-config --cxxflags --libs) -lHepMC3

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

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
  const double sqrtS = env_d("EEMUMU_SQRT_S", 250.0);
  const int nEvents = env_i("EEMUMU_N", 20000);
  const int seed = env_i("EEMUMU_SEED", 20260710);
  const char* metaPath = env_s("EEMUMU_META", "/tmp/meta.dat");
  const char* hepmcPath = env_s("EEMUMU_HEPMC", "/tmp/events.hepmc");

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

  // HepMC3 ascii3 writer (Delphes 3.5.0 DelphesHepMC3 reads this format).
  Pythia8::Pythia8ToHepMC toHepMC(hepmcPath);

  long nPrimary = 0;  // events with a primary (status-23) mu- : gen cross-check.
  for (int iE = 0; iE < nEvents; ++iE) {
    if (!pythia.next()) continue;
    // Every event goes to HepMC (Delphes sees the full sample, incl. FSR).
    toHepMC.writeNextEvent(pythia);
    // Count the primary hard-process mu- (status 23, id 13) -- an independent
    // yardstick for the macro's status-1/|p|-cut truth (which cannot use status 23,
    // as it is not preserved through the HepMC round-trip).
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (p.id() == 13 && p.statusAbs() == 23) {
        ++nPrimary;
        break;
      }
    }
  }

  const double sigma_mb = pythia.info.sigmaGen();  // mb, all flavours
  const double sigma_err_mb = pythia.info.sigmaErr();
  const double version = pythia.parm("Pythia:versionNumber");

  std::ofstream fh(metaPath);
  fh << "# process=ee->gmZ->ffbar sqrt_s_GeV=" << sqrtS << " n_generated=" << nEvents
     << " n_primary_mu=" << nPrimary << " sigma_mb=" << sigma_mb
     << " sigma_err_mb=" << sigma_err_mb << " pythia_version=" << version << "\n";

  std::cerr << "wrote HepMC3 to " << hepmcPath << " (" << nPrimary
            << " primary mu- in " << nEvents << " events); Pythia sigma = "
            << sigma_mb * 1e6 << " +/- " << sigma_err_mb * 1e6 << " nb\n";
  return 0;
}
