// Produce the generator-level (truth) AND detector-level (reco) cos(theta_mu-)
// distributions from a Delphes ROOT file -- runs INSIDE the
// scailfin/delphes-python-centos container (Delphes 3.5.0 + ROOT) via
//   ROOT_INCLUDE_PATH=/usr/local/venv/include \
//   root -l -b -q 'extract_reco.C("in.root","truth.dat","reco.dat",100.0)'
//
// BOTH distributions come from the SAME events and the SAME Delphes file -- truth
// from the generator "Particle" branch, reco from the "Muon" branch -- so the two
// populations are identical up to detector response. That is the whole point of a
// fast-sim demonstration: what the detector does to the truth. This mirrors the
// pipeline's "compute in the container, plot on the host" split (no ROOT/uproot
// dependency on the Windows host; the host analyze.py just reads two flat .dats).
//
// SIGNAL ISOLATION -- the subtle bit. The all-flavour ffbar sample also makes muons
// from tau -> mu and heavy-flavour (b/c) decays, so "any mu-" would let reco EXCEED
// truth (a detector must not *add* muons). Two facts shape the fix:
//   * Pythia's hard-outgoing status 23 is NOT preserved through the HepMC round-trip
//     (FSR replaces it with status 51/52 copies + a status-1 final), so it cannot
//     tag the signal here.
//   * The signal mu- is monochromatic at |p| ~ beam energy (~125 GeV at 250 GeV),
//     at ALL polar angles, while tau/heavy-flavour muons are soft. The |p| spectrum
//     of status-1 mu- is therefore bimodal (a spike at ~125 and a soft tail) with a
//     wide empty valley; |p| > 100 GeV sits in that valley.
// So we select signal by an angle-neutral |p| = sqrt(px^2+py^2+pz^2) > PMIN cut
// applied identically to truth and reco. Crucially |p| (not pT) is angle-neutral:
// the signal is ~125 GeV at every cos(theta), so the cut cannot manufacture a
// forward edge -- the only edge left is the detector's |eta| < 2.4 acceptance. For
// reco, |p| = PT * cosh(eta) exactly. Result: reco is a subset of truth,
// reco/truth = acceptance x efficiency <= 1, and reco vanishes beyond
// |cos theta| = tanh(2.4) = 0.984 -- the proof the detector is live.
//
// cos(theta): truth from Pz/|p| (true polar angle); reco from tanh(Muon.Eta)
// (Delphes stores pseudorapidity, eta = artanh(cos theta), exact for these
// ultra-relativistic muons). The ILD card smears muon *momentum* but keeps
// excellent angular resolution, so the visible angular detector effect is
// acceptance/efficiency, not angular smearing.

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#endif

#include <cmath>
#include <fstream>
#include <iostream>

void extract_reco(const char* inFile, const char* truthOut, const char* recoOut,
                  double pMin = 100.0) {
  gSystem->Load("libDelphes");

  TChain chain("Delphes");
  chain.Add(inFile);
  ExRootTreeReader* reader = new ExRootTreeReader(&chain);
  const Long64_t nEntries = reader->GetEntries();
  TClonesArray* branchParticle = reader->UseBranch("Particle");
  TClonesArray* branchMuon = reader->UseBranch("Muon");

  std::ofstream ft(truthOut);
  std::ofstream fr(recoOut);
  ft << "# level=truth source=GenParticle status1 p_min_GeV=" << pMin
     << " n_entries=" << nEntries << "\n";
  fr << "# level=reco detector=ILD p_min_GeV=" << pMin
     << " n_entries=" << nEntries << "\n";

  long nTruth = 0, nReco = 0;
  for (Long64_t i = 0; i < nEntries; ++i) {
    reader->ReadEntry(i);

    // TRUTH: final-state (status 1) mu- with |p| > pMin (the ~125 GeV signal muon).
    for (int j = 0; j < branchParticle->GetEntries(); ++j) {
      const GenParticle* g = static_cast<GenParticle*>(branchParticle->At(j));
      if (g->PID != 13 || g->Status != 1) continue;
      const double p = std::sqrt(g->Px * g->Px + g->Py * g->Py + g->Pz * g->Pz);
      if (p > pMin) {
        ft << (g->Pz / p) << "\n";
        ++nTruth;
      }
    }

    // RECO: reconstructed mu- with |p| = PT*cosh(eta) > pMin (same signal cut).
    for (int j = 0; j < branchMuon->GetEntries(); ++j) {
      const Muon* mu = static_cast<Muon*>(branchMuon->At(j));
      if (mu->Charge != -1) continue;
      const double p = mu->PT * std::cosh(mu->Eta);
      if (p > pMin) {
        fr << std::tanh(mu->Eta) << "\n";
        ++nReco;
      }
    }
  }
  std::cerr << "truth mu- (status1, |p|>" << pMin << "): " << nTruth
            << "   reco mu- (|p|>" << pMin << "): " << nReco
            << "   acc*eff = " << (nTruth ? double(nReco) / double(nTruth) : 0.0)
            << "\n";
}
