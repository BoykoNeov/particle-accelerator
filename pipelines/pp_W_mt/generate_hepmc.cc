// Generate  p p -> W -> mu nu  with Pythia8 at sqrt(s) = 13 TeV using a real
// LHAPDF6 proton PDF, and write a full-event HepMC3 file (+ a small meta header).
// Compiled and run INSIDE the hepstore/rivet-pythia container (Pythia8 8.3 +
// HepMC3 + LHAPDF6). This is the generation stage of milestone E1 -- the W-mass
// Jacobian edge -- and the charged-current sibling of the neutral-current
// Drell-Yan chain in ../pp_mumu_drellyan/.
//
// WHAT MAKES THIS DIFFERENT FROM THE Z CHAIN: THE NEUTRINO ESCAPES. In
// pp -> gamma*/Z -> mu+mu- both decay products are measured, so the invariant
// mass m(mumu) is reconstructible and the observable is a resonance PEAK. Here
// one product is a neutrino: it leaves no detector signal at all, and its p_z is
// not recoverable even in principle (the longitudinal boost of the qqbar system
// is unknown). So there is NO invariant mass to build -- the observable is the
// TRANSVERSE mass
//
//     m_T^2 = 2 p_T^mu p_T^nu (1 - cos(dphi)),
//
// whose distribution has a JACOBIAN EDGE at m_T = M_W. See
// docs/CONVENTIONS.md -> "Transverse mass and the W Jacobian edge".
//
// BOTH W CHARGES ARE KEPT. The edge sits at M_W for W+ and W- alike, and pp
// collisions produce more W+ than W- (the proton's valence uud favours
// u dbar -> W+). Splitting by charge would only cost statistics on a
// charge-independent observable; the asymmetry itself is not an E1 deliverable.
//
// NO MASS WINDOW. The Z chain set PhaseSpace:mHatMin/Max = 60..120 to avoid the
// divergent low-mass photon pole. There is no such pole here (the charged current
// has no photon-exchange piece), and -- more importantly -- a window would be
// ACTIVELY HARMFUL: it would impose a hard cutoff near the very edge this
// milestone measures, manufacturing an artificial one. The W's Breit-Wigner
// (Gamma_W ~ 2.09 GeV) must be allowed to smear the edge on its own; off-shell
// events with m(mu nu) > M_W are real physics and are deliberately kept.
//
// ISR/FSR stay ON. ISR gives the W a recoil p_T, which is precisely why m_T is
// the W-mass observable rather than the lepton p_T: the m_T edge is insensitive
// to that recoil at first order, the p_T^mu peak (at M_W/2) is not.
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
  const double sqrtS = env_d("W_SQRT_S", 13000.0);
  const int nEvents = env_i("W_N", 20000);
  const int seed = env_i("W_SEED", 20260720);
  const char* pdfSet = env_s("W_PDF_SET", "NNPDF31_lo_as_0118");
  const char* metaPath = env_s("W_META", "/tmp/meta.dat");
  const char* hepmcPath = env_s("W_HEPMC", "/tmp/events.hepmc");

  Pythia pythia;
  pythia.readString("Beams:idA = 2212");  // proton
  pythia.readString("Beams:idB = 2212");  // proton
  pythia.settings.parm("Beams:eCM", sqrtS);

  // Real LHAPDF6 proton PDF, member 0 -- an LO set to match Pythia's LO matrix
  // element (same reasoning as the DY chain; see that README).
  pythia.settings.word("PDF:pSet", std::string("LHAPDF6:") + pdfSet + "/0");

  // Charged-current single-boson production: f fbar' -> W+-.
  pythia.readString("WeakSingleBoson:ffbar2W = on");

  // Force W -> mu nu_mu, BOTH charges. onIfAny matches a channel if ANY listed
  // |id| appears in it, and Pythia applies the charge conjugate to W- itself, so
  // this single pair of lines covers W+ -> mu+ nu_mu and W- -> mu- nubar_mu while
  // removing the e/tau channels (tau -> mu would otherwise contaminate the sample
  // with a SOFTER muon and a THREE-neutrino missing-momentum vector, both of which
  // smear the edge for reasons that have nothing to do with the W mass).
  pythia.readString("24:onMode = off");
  pythia.readString("24:onIfAny = 13 14");

  pythia.readString("Random:setSeed = on");
  pythia.settings.mode("Random:seed", seed);
  pythia.readString("Print:quiet = on");
  pythia.readString("Next:numberCount = 0");

  if (!pythia.init()) {
    std::cerr << "Pythia failed to initialise (is the LHAPDF set '" << pdfSet
              << "' installed?)\n";
    return 1;
  }

  Pythia8::Pythia8ToHepMC toHepMC(hepmcPath);

  // Sanity counters. nHardMu counts hard-process (status 23) muons of either
  // charge: since the decay is FORCED, essentially every generated event has one,
  // so a number far below n_generated means the forced decay did not fire.
  long nHardMu = 0, nPlus = 0, nMinus = 0;
  for (int iE = 0; iE < nEvents; ++iE) {
    if (!pythia.next()) continue;
    toHepMC.writeNextEvent(pythia);  // Delphes sees the full event (incl. ISR/FSR)
    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      if (std::abs(p.id()) == 13 && p.statusAbs() == 23) ++nHardMu;
      if (p.id() == 24 && p.statusAbs() == 22) ++nPlus;
      if (p.id() == -24 && p.statusAbs() == 22) ++nMinus;
    }
  }

  const double sigma_mb = pythia.info.sigmaGen();  // mb, W production x BR(W->munu)
  const double sigma_err_mb = pythia.info.sigmaErr();
  const double version = pythia.parm("Pythia:versionNumber");
  // Read the W mass and width back OUT of Pythia rather than echoing a remembered
  // PDG number: the gate compares the measured edge against the mass the generator
  // ACTUALLY used, so hardcoding a constant on the analysis side would turn the
  // gate into a comparison of two remembered numbers.
  const double mW = pythia.particleData.m0(24);
  const double widthW = pythia.particleData.mWidth(24);

  std::ofstream fh(metaPath);
  fh << "# process=pp->W->munu sqrt_s_GeV=" << sqrtS << " n_generated=" << nEvents
     << " n_hard_mu=" << nHardMu << " n_wplus=" << nPlus << " n_wminus=" << nMinus
     << " pdf_set=" << pdfSet << " pdf_member=0" << " sigma_mb=" << sigma_mb
     << " sigma_err_mb=" << sigma_err_mb << " pythia_version=" << version
     << " m_w_gev=" << std::setprecision(9) << mW
     << " width_w_gev=" << widthW << "\n";

  std::cerr << "wrote HepMC3 to " << hepmcPath << " (" << nHardMu << " hard mu in "
            << nEvents << " events; W+ " << nPlus << " / W- " << nMinus
            << "); PDF=" << pdfSet << "; M_W = " << mW << " GeV, Gamma_W = " << widthW
            << " GeV; sigma = " << sigma_mb * 1e6 << " +/- " << sigma_err_mb * 1e6
            << " nb (W production x BR(W->mu nu))\n";
  return 0;
}
