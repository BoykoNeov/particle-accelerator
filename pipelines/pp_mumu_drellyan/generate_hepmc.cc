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
#include <iomanip>
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
  // Generator-level truth dump carrying the TRUE incoming-quark direction (see
  // below) -- the reference the pp sign(Q_z) proxy is diluted against for A_FB.
  const char* truthGenPath = env_s("DY_TRUTHGEN", "/tmp/truth_gen.dat");

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

  // --- the weak mixing angle (milestone A2) --------------------------------
  // A_FB's sensitivity runs through the *fermion vector coupling*
  // g_V^f = T3_f - 2 Q_f sin^2(theta_W), which is built from the EFFECTIVE angle
  // (`sin2thetaWbar`), not the on-shell one (`sin2thetaW`, which fixes the W/Z
  // mass relation). Pythia keeps them as separate parameters. Leaving them at
  // their defaults would make "recover the value Pythia was configured with" an
  // ambiguous gate -- and inviting the analysis to hardcode a remembered default
  // is exactly the failure mode this project guards against. So: set BOTH to the
  // same explicit value, and emit both into meta.dat as the unambiguous truth.
  const double sin2ThetaW = env_d("DY_SIN2THETAW", 0.2312);
  pythia.settings.parm("StandardModel:sin2thetaW", sin2ThetaW);
  pythia.settings.parm("StandardModel:sin2thetaWbar", sin2ThetaW);

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

  // Generator-level truth dump for the A_FB *dilution* demonstration. Per event:
  //   quark_pz_sign  Em pxm pym pzm  Ep pxp pyp pzp
  // The quark_pz_sign is the sign of p_z of the TRUE incoming quark of the hard
  // q qbar -> gamma*/Z process (status -21, i.e. statusAbs()==21, id in 1..6). In
  // pp we normally do NOT know this -- the analysis proxies it by sign(Q_z), the
  // di-lepton boost -- so comparing A_FB with this true sign vs the proxy sign is
  // exactly the pp dilution (worst at central rapidity). mu- is id +13, mu+ id -13.
  std::ofstream ftg(truthGenPath);
  ftg << "# level=truth_gen source=Pythia observable=quark_pz_sign+mu-_mu+_fourvectors"
      << " cols=qsign,Em,pxm,pym,pzm,Ep,pxp,pyp,pzp\n";
  long nTruthGen = 0;

  // Sanity counter: events carrying a hard-process (status 23) mu-. Because we
  // FORCE 23 -> mu+ mu-, every successfully generated event has one, so this is
  // just a "the forced decay fired everywhere" check (expect == n successful
  // next()), NOT the independent truth yardstick it was in the leptonic chain
  // (where the process summed all flavours). The real cross-check here is the
  // cross section vs the known LHC value -- see the README.
  long nHardMu = 0;
  for (int iE = 0; iE < nEvents; ++iE) {
    if (!pythia.next()) continue;
    toHepMC.writeNextEvent(pythia);  // Delphes sees the full event (incl. ISR/FSR).

    // True incoming-quark p_z sign + leading OS status-1 muon pair, from Pythia's
    // own record (the parton-level truth; no HepMC/Delphes round-trip).
    double quarkPz = 0.0;
    int muMinus = -1, muPlus = -1;  // event indices of the leading mu-/mu+
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (p.statusAbs() == 21 && p.id() >= 1 && p.id() <= 6) {
        quarkPz = p.pz();  // the hard incoming quark (not the antiquark)
      }
      if (p.id() == 13 && p.statusAbs() == 23) ++nHardMu;
      if (!p.isFinal()) continue;
      if (p.id() == 13 && (muMinus < 0 || p.pT() > pythia.event[muMinus].pT()))
        muMinus = i;  // mu- is id +13
      else if (p.id() == -13 && (muPlus < 0 || p.pT() > pythia.event[muPlus].pT()))
        muPlus = i;  // mu+ is id -13
    }
    if (quarkPz != 0.0 && muMinus >= 0 && muPlus >= 0) {
      const Particle& m = pythia.event[muMinus];
      const Particle& p = pythia.event[muPlus];
      ftg << std::setprecision(9) << (quarkPz > 0 ? 1 : -1) << " " << m.e() << " "
          << m.px() << " " << m.py() << " " << m.pz() << "  " << p.e() << " " << p.px()
          << " " << p.py() << " " << p.pz() << "\n";
      ++nTruthGen;
    }
  }

  const double sigma_mb = pythia.info.sigmaGen();  // mb, DY x BR(Z->mumu) in window
  const double sigma_err_mb = pythia.info.sigmaErr();
  const double version = pythia.parm("Pythia:versionNumber");
  // Read the mixing angles back OUT of Pythia rather than echoing the input, so
  // meta.dat records what the generator actually ran with. Both are emitted; the
  // A_FB fit recovers the effective one (`sin2thetaWbar`).
  const double s2wOnShell = pythia.parm("StandardModel:sin2thetaW");
  const double s2wEffective = pythia.parm("StandardModel:sin2thetaWbar");

  std::ofstream fh(metaPath);
  fh << "# process=pp->gmZ->mumu sqrt_s_GeV=" << sqrtS << " n_generated=" << nEvents
     << " n_hard_mu=" << nHardMu << " mhat_min_GeV=" << mMin
     << " mhat_max_GeV=" << mMax << " pdf_set=" << pdfSet << " pdf_member=0"
     << " sigma_mb=" << sigma_mb << " sigma_err_mb=" << sigma_err_mb
     << " pythia_version=" << version << " sin2thetaw=" << s2wOnShell
     << " sin2thetawbar=" << s2wEffective << "\n";

  std::cerr << "wrote HepMC3 to " << hepmcPath << " (" << nHardMu
            << " hard mu- in " << nEvents << " events); PDF=" << pdfSet
            << "; Pythia sigma = " << sigma_mb * 1e6 << " +/- " << sigma_err_mb * 1e6
            << " nb (DY x BR(Z->mumu), " << mMin << "<m<" << mMax << ")\n";
  std::cerr << "wrote generator truth (quark dir + mu pair) for " << nTruthGen
            << " events to " << truthGenPath << "\n";
  return 0;
}
