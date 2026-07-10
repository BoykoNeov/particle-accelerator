// Generate hadronic Drell-Yan  p p -> gamma*/Z -> mu+ mu-  with Pythia8 at
// sqrt(s) = 13 TeV, using a *real LHAPDF6 proton PDF*, and write a full-event
// HepMC3 file (+ a small meta header). Compiled and run INSIDE the
// hepstore/rivet-pythia container (Pythia8 8.3 + HepMC3 + LHAPDF6). This is the
// generation stage of the *hadronic* extension of Phase 2 -- the leptonic chains
// (ee_mumu_pythia / ee_mumu_delphes) had a fixed partonic sqrt(s); here the PDFs
// make the initial-state momentum -- and therefore the partonic mHat -- a
// distribution, which is the whole point of "with real PDFs".
//
// WHY THE 2->1 RESONANT PROCESS WORKS HERE (it did NOT leptonically). The leptonic
// chains used the 2->2 continuum ffbar2ffbar(s:gmZ) because the 2->1 resonant
// ffbar2gmZ *underflows to zero* at a fixed partonic sqrt(s) far below the Z (its
// Breit-Wigner integrates over a delta-function mHat). With protons the PDFs spread
// the partonic mHat across a continuum, so the 2->1 resonant process is exactly the
// right tool: WeakSingleBoson:ffbar2gmZ = q qbar -> gamma*/Z, the textbook
// Drell-Yan production channel.
//
// CLEAN DIMUON SAMPLE BY FORCED DECAY (no |p| cut needed, unlike the leptonic
// Delphes chain). Because this is a resonance process we can force the boson decay
// 23 -> mu+ mu- (23:onMode/onIfMatch). That removes tau->mu and heavy-flavour muon
// contamination *at the source*, so the downstream Delphes macro selects the signal
// simply as the leading opposite-sign muon pair -- no monochromatic-|p| trick.
// NB: forcing the decay means Pythia's reported sigmaGen() is the production cross
// section *times* BR(Z->mumu) (verified empirically against the known LHC value in
// the README), i.e. the mu-channel Drell-Yan cross section in the generated window.
//
// MASS WINDOW. PhaseSpace:mHatMin/Max = 60..120 GeV -- the standard fiducial
// Z-peak window, matching the measured LHC sigma(pp->Z->ll, 60<m<120) so the
// meta.dat cross section is a *meaningful* number, not a divergent low-mass
// (photon-pole) integral.
//
// ISR/FSR stay ON (physical for hadronic DY). We deliberately do NOT set
// PDF:lepton = off -- that was a leptonic-beam ISR toggle, irrelevant to protons.
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
  const double sqrtS = env_d("DY_SQRT_S", 13000.0);
  const int nEvents = env_i("DY_N", 20000);
  const int seed = env_i("DY_SEED", 20260710);
  const double mMin = env_d("DY_MHAT_MIN", 60.0);
  const double mMax = env_d("DY_MHAT_MAX", 120.0);
  const char* pdfSet = env_s("DY_PDF_SET", "NNPDF31_lo_as_0118");
  const char* metaPath = env_s("DY_META", "/tmp/meta.dat");
  const char* hepmcPath = env_s("DY_HEPMC", "/tmp/events.hepmc");

  Pythia pythia;
  pythia.readString("Beams:idA = 2212");  // proton
  pythia.readString("Beams:idB = 2212");  // proton
  pythia.settings.parm("Beams:eCM", sqrtS);

  // Real LHAPDF6 proton PDF, member 0. An LO set to match Pythia's LO matrix
  // element (see README). If the set is missing (not downloaded), Pythia init
  // fails -- the pipeline runs `lhapdf get` first and reports a clean error.
  pythia.settings.word("PDF:pSet", std::string("LHAPDF6:") + pdfSet + "/0");

  // Drell-Yan production: q qbar -> gamma*/Z.
  pythia.readString("WeakSingleBoson:ffbar2gmZ = on");
  // Force the boson to mu+ mu- -> a clean dimuon sample at the source.
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfMatch = 13 -13");

  // Fiducial Z-peak mass window (avoids the divergent photon pole at low mass).
  pythia.settings.parm("PhaseSpace:mHatMin", mMin);
  pythia.settings.parm("PhaseSpace:mHatMax", mMax);

  pythia.readString("Random:setSeed = on");
  pythia.settings.mode("Random:seed", seed);
  pythia.readString("Print:quiet = on");
  pythia.readString("Next:numberCount = 0");

  if (!pythia.init()) {
    std::cerr << "Pythia failed to initialise (is the LHAPDF set '" << pdfSet
              << "' installed?)\n";
    return 1;
  }

  // HepMC3 ascii3 writer (Delphes 3.5.0 DelphesHepMC3 reads this format).
  Pythia8::Pythia8ToHepMC toHepMC(hepmcPath);

  long nDimuon = 0;  // events with a hard-process (status 23) mu- : gen cross-check.
  for (int iE = 0; iE < nEvents; ++iE) {
    if (!pythia.next()) continue;
    toHepMC.writeNextEvent(pythia);  // Delphes sees the full event (incl. ISR/FSR).
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (p.id() == 13 && p.statusAbs() == 23) {
        ++nDimuon;
        break;
      }
    }
  }

  const double sigma_mb = pythia.info.sigmaGen();  // mb, DY x BR(Z->mumu) in window
  const double sigma_err_mb = pythia.info.sigmaErr();
  const double version = pythia.parm("Pythia:versionNumber");

  std::ofstream fh(metaPath);
  fh << "# process=pp->gmZ->mumu sqrt_s_GeV=" << sqrtS << " n_generated=" << nEvents
     << " n_primary_mu=" << nDimuon << " mhat_min_GeV=" << mMin
     << " mhat_max_GeV=" << mMax << " pdf_set=" << pdfSet << " pdf_member=0"
     << " sigma_mb=" << sigma_mb << " sigma_err_mb=" << sigma_err_mb
     << " pythia_version=" << version << "\n";

  std::cerr << "wrote HepMC3 to " << hepmcPath << " (" << nDimuon
            << " hard mu- in " << nEvents << " events); PDF=" << pdfSet
            << "; Pythia sigma = " << sigma_mb * 1e6 << " +/- " << sigma_err_mb * 1e6
            << " nb (DY x BR(Z->mumu), " << mMin << "<m<" << mMax << ")\n";
  return 0;
}
