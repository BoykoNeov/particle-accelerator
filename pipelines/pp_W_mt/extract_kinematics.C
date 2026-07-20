// Dump the generator-level (truth) AND detector-level (reco) kinematics needed for
// the W transverse mass -- the muon four-vector plus the missing-transverse-momentum
// vector, one line per event -- from a Delphes ROOT file. Runs INSIDE the
// scailfin/delphes-python-centos container (Delphes 3.5.0 + ROOT):
//   ROOT_INCLUDE_PATH=/usr/local/venv/include \
//   root -l -b -q 'extract_kinematics.C("in.root","truth_kin.dat","reco_kin.dat")'
//
// WHY RAW VECTORS, NOT m_T. Exactly as in the Drell-Yan chain: the container only
// EXTRACTS, and all physics is computed on the host by the single *tested*
// implementation accsim.events.transverse_mass (analytic gate:
// tests/analytic/test_transverse_mass.py). Recomputing m_T in untested C++ here
// would duplicate a (1 - cos dphi) that no test ever sees.
//
// THE TRUTH-vs-RECO SEAM IS THE NEUTRINO PROXY. Truth uses the generated
// invisible momentum; reco uses the detector's MissingET. That substitution -- not
// the muon, which CMS measures well -- is what rounds the Jacobian edge.
//
// MUONS ARE INSIDE Delphes' MissingET, VERIFIED IN THE CARD. MissingET is the
// Merger over EFlowMerger/eflow <- HCal/eflowTracks <- TrackMerger, and TrackMerger
// takes MuonMomentumSmearing/muons (delphes_card_CMS.tcl line ~201). Had the muon
// been excluded, MET would track the hadronic recoil instead of the neutrino and
// every reco m_T would be meaningless -- so this was checked, not assumed.
//
// THE GenMissingET SIGN IS PINNED BY DATA, NOT BY MEMORY. Delphes' Merger negates
// its vector sum to form a "missing" momentum, and GenMissingET's input is the
// NEUTRINO list itself (PdgCodeFilter on |pid| in {12,14,16}) rather than the
// visible particles. Whether the resulting vector therefore points ALONG the
// neutrino or OPPOSITE to it decides dphi by a full pi -- which flips
// (1 - cos dphi) between ~0 and ~2 and would wreck m_T. Rather than trust a
// remembered convention, this macro emits BOTH the GenMissingET vector AND the
// directly summed truth neutrino four-vector, and analyze.py measures the angle
// between them and pins the convention empirically (refusing to run if they agree
// with neither 0 nor pi).
//
// SIGNAL SELECTION -- the leading muon (highest pT), of either charge, applied
// identically to truth and reco. The generator forced W -> mu nu, so the only
// prompt muon is the signal one; both W charges are kept (the edge is at M_W for
// each). No opposite-sign pairing exists here, unlike the Z chain.
//
// OUTPUT (whitespace-separated, one line per event that has a muon):
//   truth: E px py pz   genmet_pt genmet_phi   nu_pt nu_phi   charge
//   reco:  E px py pz   met_pt    met_phi      0     0        charge
// The two files are INDEPENDENT subsets (a line is written only when THAT level
// has a muon), so line i of truth is NOT line i of reco -- analyze.py histograms
// each independently and never joins on line index, exactly as the DY macro does.

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#endif

#include "TLorentzVector.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

static const double MU_MASS = 0.1056583745;  // GeV (PDG)

void extract_kinematics(const char* inFile, const char* truthOut, const char* recoOut) {
  gSystem->Load("libDelphes");

  TChain chain("Delphes");
  chain.Add(inFile);
  ExRootTreeReader* reader = new ExRootTreeReader(&chain);
  const Long64_t nEntries = reader->GetEntries();
  TClonesArray* branchParticle = reader->UseBranch("Particle");
  TClonesArray* branchMuon = reader->UseBranch("Muon");
  TClonesArray* branchMET = reader->UseBranch("MissingET");
  TClonesArray* branchGenMET = reader->UseBranch("GenMissingET");

  std::ofstream ft(truthOut);
  std::ofstream fr(recoOut);
  ft << "# level=truth source=GenParticle+GenMissingET observable=muon+missing"
     << " cols=E,px,py,pz,met_pt,met_phi,nu_pt,nu_phi,charge n_entries=" << nEntries
     << "\n";
  fr << "# level=reco detector=CMS observable=muon+MissingET"
     << " cols=E,px,py,pz,met_pt,met_phi,nu_pt,nu_phi,charge n_entries=" << nEntries
     << "\n";

  long nTruth = 0, nReco = 0;
  for (Long64_t i = 0; i < nEntries; ++i) {
    reader->ReadEntry(i);

    // ---- TRUTH: leading status-1 muon + generated missing momentum -----------
    {
      TLorentzVector mu;
      bool has = false;
      double charge = 0.0;
      // Direct neutrino sum, kept as the independent cross-check of GenMissingET.
      double nuPx = 0.0, nuPy = 0.0;
      for (int j = 0; j < branchParticle->GetEntries(); ++j) {
        const GenParticle* g = static_cast<GenParticle*>(branchParticle->At(j));
        if (g->Status != 1) continue;
        const int apid = std::abs(g->PID);
        if (apid == 13) {
          TLorentzVector v(g->Px, g->Py, g->Pz, g->E);
          if (!has || v.Pt() > mu.Pt()) {
            mu = v;
            has = true;
            charge = (g->PID == 13) ? -1.0 : +1.0;  // mu- is PID +13
          }
        } else if (apid == 12 || apid == 14 || apid == 16) {
          nuPx += g->Px;
          nuPy += g->Py;
        }
      }
      if (has) {
        double gmetPt = 0.0, gmetPhi = 0.0;
        if (branchGenMET->GetEntries() > 0) {
          const MissingET* m = static_cast<MissingET*>(branchGenMET->At(0));
          gmetPt = m->MET;
          gmetPhi = m->Phi;
        }
        ft << std::setprecision(9) << mu.E() << " " << mu.Px() << " " << mu.Py() << " "
           << mu.Pz() << "  " << gmetPt << " " << gmetPhi << "  "
           << std::sqrt(nuPx * nuPx + nuPy * nuPy) << " " << std::atan2(nuPy, nuPx)
           << "  " << charge << "\n";
        ++nTruth;
      }
    }

    // ---- RECO: leading reconstructed muon + MissingET (same events) ----------
    {
      TLorentzVector mu;
      bool has = false;
      double charge = 0.0;
      for (int j = 0; j < branchMuon->GetEntries(); ++j) {
        const Muon* m = static_cast<Muon*>(branchMuon->At(j));
        TLorentzVector v;
        v.SetPtEtaPhiM(m->PT, m->Eta, m->Phi, MU_MASS);
        if (!has || v.Pt() > mu.Pt()) {
          mu = v;
          has = true;
          charge = m->Charge;
        }
      }
      if (has && branchMET->GetEntries() > 0) {
        const MissingET* m = static_cast<MissingET*>(branchMET->At(0));
        fr << std::setprecision(9) << mu.E() << " " << mu.Px() << " " << mu.Py() << " "
           << mu.Pz() << "  " << m->MET << " " << m->Phi << "  0 0  " << charge << "\n";
        ++nReco;
      }
    }
  }
  std::cerr << "truth muons: " << nTruth << "   reco muons: " << nReco
            << "   reco/truth = " << (nTruth ? double(nReco) / double(nTruth) : 0.0)
            << "\n";
}
