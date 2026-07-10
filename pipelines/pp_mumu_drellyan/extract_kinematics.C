// Dump the generator-level (truth) AND detector-level (reco) di-muon KINEMATICS --
// the mu- and mu+ four-vectors, one line per event -- from a Delphes ROOT file.
// Runs INSIDE the scailfin/delphes-python-centos container (Delphes 3.5.0 + ROOT):
//   ROOT_INCLUDE_PATH=/usr/local/venv/include \
//   root -l -b -q 'extract_kinematics.C("in.root","truth_kin.dat","reco_kin.dat")'
//
// WHY FOUR-VECTORS, NOT JUST THE MASS. Both the di-muon invariant-mass spectrum
// (the Z peak) AND the Collins-Soper cos(theta*) / forward-backward asymmetry A_FB
// are functions of the SAME two four-vectors. So this macro emits the raw
// four-vectors and ALL physics -- m(mumu) and cos(theta*)_CS -- is computed on the
// host by the *single tested* implementation in accsim.events.kinematics
// (collins_soper_costheta), rather than duplicating a sign-error-prone frame
// transform in untested C++ here. The container just extracts; Python does physics.
//
// BOTH levels come from the SAME events and the SAME Delphes file -- truth from the
// generator "Particle" branch (status-1, post-FSR muons), reco from the "Muon"
// branch -- so they are one population up to detector response, and Delphes
// preserves HepMC event order, so line i of truth and line i of reco are the same
// event when both have a pair (a per-event flag column records which).
//
// SIGNAL SELECTION -- the leading opposite-sign muon pair (highest-pT mu+ and
// highest-pT mu-), applied identically to truth and reco. The generator forced the
// boson decay 23 -> mu+ mu-, so the only prompt muons are the signal pair -- no
// tau->mu contamination, hence no monochromatic-|p| cut. mu- carries PID +13,
// mu+ carries PID -13 (kept exactly -- one flip inverts A_FB).
//
// OUTPUT (whitespace-separated, one physics line per event that has an OS pair):
//   Em pxm pym pzm  Ep pxp pyp pzp
// mu- four-vector first, then mu+. Events without an OS pair emit no line.

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#endif

#include "TLorentzVector.h"

#include <fstream>
#include <iomanip>
#include <iostream>

static const double MU_MASS = 0.1056583745;  // GeV (PDG)

// Keep the highest-pT four-vector of a given charge sign seen so far.
static void consider(TLorentzVector& lead, bool& found, const TLorentzVector& cand) {
  if (!found || cand.Pt() > lead.Pt()) {
    lead = cand;
    found = true;
  }
}

static void write_pair(std::ofstream& fh, const TLorentzVector& minus,
                       const TLorentzVector& plus) {
  fh << std::setprecision(9) << minus.E() << " " << minus.Px() << " " << minus.Py()
     << " " << minus.Pz() << "  " << plus.E() << " " << plus.Px() << " " << plus.Py()
     << " " << plus.Pz() << "\n";
}

void extract_kinematics(const char* inFile, const char* truthOut, const char* recoOut) {
  gSystem->Load("libDelphes");

  TChain chain("Delphes");
  chain.Add(inFile);
  ExRootTreeReader* reader = new ExRootTreeReader(&chain);
  const Long64_t nEntries = reader->GetEntries();
  TClonesArray* branchParticle = reader->UseBranch("Particle");
  TClonesArray* branchMuon = reader->UseBranch("Muon");

  std::ofstream ft(truthOut);
  std::ofstream fr(recoOut);
  ft << "# level=truth source=GenParticle status1 observable=mu-_mu+_fourvectors"
     << " cols=Em,pxm,pym,pzm,Ep,pxp,pyp,pzp n_entries=" << nEntries << "\n";
  fr << "# level=reco detector=CMS observable=mu-_mu+_fourvectors"
     << " cols=Em,pxm,pym,pzm,Ep,pxp,pyp,pzp n_entries=" << nEntries << "\n";

  long nTruth = 0, nReco = 0;
  for (Long64_t i = 0; i < nEntries; ++i) {
    reader->ReadEntry(i);

    // TRUTH: leading OS pair among final-state (status 1) muons. mu- is PID +13.
    {
      TLorentzVector minus, plus;
      bool hasMinus = false, hasPlus = false;
      for (int j = 0; j < branchParticle->GetEntries(); ++j) {
        const GenParticle* g = static_cast<GenParticle*>(branchParticle->At(j));
        if (std::abs(g->PID) != 13 || g->Status != 1) continue;
        TLorentzVector v(g->Px, g->Py, g->Pz, g->E);
        if (g->PID == 13)
          consider(minus, hasMinus, v);  // mu- has PID +13
        else
          consider(plus, hasPlus, v);  // mu+ has PID -13
      }
      if (hasMinus && hasPlus) {
        write_pair(ft, minus, plus);
        ++nTruth;
      }
    }

    // RECO: leading OS pair among reconstructed muons (same events).
    {
      TLorentzVector minus, plus;
      bool hasMinus = false, hasPlus = false;
      for (int j = 0; j < branchMuon->GetEntries(); ++j) {
        const Muon* mu = static_cast<Muon*>(branchMuon->At(j));
        TLorentzVector v;
        v.SetPtEtaPhiM(mu->PT, mu->Eta, mu->Phi, MU_MASS);
        if (mu->Charge < 0)
          consider(minus, hasMinus, v);  // mu- has negative charge
        else
          consider(plus, hasPlus, v);
      }
      if (hasMinus && hasPlus) {
        write_pair(fr, minus, plus);
        ++nReco;
      }
    }
  }
  std::cerr << "truth pairs: " << nTruth << "   reco pairs: " << nReco
            << "   reco/truth = " << (nTruth ? double(nReco) / double(nTruth) : 0.0)
            << "\n";
}
