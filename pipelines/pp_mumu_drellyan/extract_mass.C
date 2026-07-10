// Produce the generator-level (truth) AND detector-level (reco) di-muon INVARIANT
// MASS m(mu+ mu-) from a Delphes ROOT file -- runs INSIDE the
// scailfin/delphes-python-centos container (Delphes 3.5.0 + ROOT) via
//   ROOT_INCLUDE_PATH=/usr/local/venv/include \
//   root -l -b -q 'extract_mass.C("in.root","truth.dat","reco.dat")'
//
// BOTH distributions come from the SAME events and the SAME Delphes file -- truth
// from the generator "Particle" branch, reco from the "Muon" branch -- so they are
// one population up to detector response. That is the deliverable of a fast-sim
// demonstration: what the CMS detector does to the truth Drell-Yan Z peak. The
// split (compute in-container, plot on the host) mirrors the leptonic Delphes
// chain: no ROOT/uproot dependency on the Windows host; analyze.py reads flat .dats.
//
// SIGNAL SELECTION -- simpler than the leptonic chain. The generator forced the
// boson decay 23 -> mu+ mu-, so the ONLY prompt muons are the signal pair; there is
// no tau->mu / heavy-flavour contamination to reject, hence NO monochromatic-|p|
// cut. We just take the LEADING opposite-sign muon pair (highest-pT mu+ and
// highest-pT mu-) -- robust when FSR occasionally yields more than two status-1
// muons. Applied identically to truth and reco.
//
// THE DETECTOR SIGNATURE IS MASS RESOLUTION, not an acceptance edge. The CMS card
// smears muon *momentum* (~1-2% per muon), which broadens the reco Z peak *wider*
// than the truth peak (whose width is the natural Gamma_Z ~ 2.49 GeV plus an
// FSR low-side radiative tail). Sharp truth peak vs broadened reco peak is the
// visible proof the detector step is live -- the parallel of the leptonic chain's
// |eta|<2.4 edge.
//
// Invariant mass via TLorentzVector: truth muons carry (Px,Py,Pz,E) directly; reco
// muons are stored as (PT, Eta, Phi), so SetPtEtaPhiM with the PDG muon mass.

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#endif

#include "TLorentzVector.h"

#include <fstream>
#include <iostream>

static const double MU_MASS = 0.1056583745;  // GeV (PDG)

// Fill `lead` with the highest-pT four-vector of the requested charge sign found so
// far; returns true if a candidate of that sign exists. `found` guards the seed.
static void consider(TLorentzVector& lead, bool& found, const TLorentzVector& cand) {
  if (!found || cand.Pt() > lead.Pt()) {
    lead = cand;
    found = true;
  }
}

void extract_mass(const char* inFile, const char* truthOut, const char* recoOut) {
  gSystem->Load("libDelphes");

  TChain chain("Delphes");
  chain.Add(inFile);
  ExRootTreeReader* reader = new ExRootTreeReader(&chain);
  const Long64_t nEntries = reader->GetEntries();
  TClonesArray* branchParticle = reader->UseBranch("Particle");
  TClonesArray* branchMuon = reader->UseBranch("Muon");

  std::ofstream ft(truthOut);
  std::ofstream fr(recoOut);
  ft << "# level=truth source=GenParticle status1 observable=m_mumu_GeV"
     << " n_entries=" << nEntries << "\n";
  fr << "# level=reco detector=CMS observable=m_mumu_GeV"
     << " n_entries=" << nEntries << "\n";

  long nTruth = 0, nReco = 0;
  for (Long64_t i = 0; i < nEntries; ++i) {
    reader->ReadEntry(i);

    // TRUTH: leading OS pair among final-state (status 1) muons.
    {
      TLorentzVector plus, minus;
      bool hasPlus = false, hasMinus = false;
      for (int j = 0; j < branchParticle->GetEntries(); ++j) {
        const GenParticle* g = static_cast<GenParticle*>(branchParticle->At(j));
        if (std::abs(g->PID) != 13 || g->Status != 1) continue;
        TLorentzVector v(g->Px, g->Py, g->Pz, g->E);
        if (g->PID == -13)
          consider(plus, hasPlus, v);  // mu+ has PID -13
        else
          consider(minus, hasMinus, v);
      }
      if (hasPlus && hasMinus) {
        ft << (plus + minus).M() << "\n";
        ++nTruth;
      }
    }

    // RECO: leading OS pair among reconstructed muons (same events).
    {
      TLorentzVector plus, minus;
      bool hasPlus = false, hasMinus = false;
      for (int j = 0; j < branchMuon->GetEntries(); ++j) {
        const Muon* mu = static_cast<Muon*>(branchMuon->At(j));
        TLorentzVector v;
        v.SetPtEtaPhiM(mu->PT, mu->Eta, mu->Phi, MU_MASS);
        if (mu->Charge > 0)
          consider(plus, hasPlus, v);
        else
          consider(minus, hasMinus, v);
      }
      if (hasPlus && hasMinus) {
        fr << (plus + minus).M() << "\n";
        ++nReco;
      }
    }
  }
  std::cerr << "truth m(mumu) pairs: " << nTruth << "   reco m(mumu) pairs: " << nReco
            << "   reco/truth = " << (nTruth ? double(nReco) / double(nTruth) : 0.0)
            << "\n";
}
