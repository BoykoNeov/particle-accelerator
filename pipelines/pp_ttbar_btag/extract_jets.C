// Dump the reconstructed jets -- kinematics, associated parton flavour, and the
// b-tag bitmask -- from a Delphes ROOT file, one line per jet.
// Runs INSIDE the scailfin/delphes-python-centos container (Delphes 3.5.0 + ROOT):
//   ROOT_INCLUDE_PATH=/usr/local/venv/include \
//   root -l -b -q 'extract_jets.C("in.root","jets.dat")'
//
// WHY THE BITMASK IS EMITTED RAW. Jet.BTag is a bit MASK, not a boolean: the
// CMS_PhaseII card runs three BTagging modules writing bits 0/1/2 (Loose /
// Medium / Tight) into the same integer. Testing `BTag == 1` would therefore
// mean "loose but NOT medium", which is not a working point at all. The mask is
// written out untouched and decoded on the host by the single tested
// implementation in accsim.events.btag (BTagWorkingPoint.tagged), whose bit
// number comes from parsing the card -- so the card, not this macro, decides
// which bit means what. The container extracts; Python does physics.
//
// WHY THE EVENT INDEX IS EMITTED. It ties each jet to the generator-level heavy
// quarks that generate_hepmc.cc dumped for the same event, which is what lets
// the host build an INDEPENDENT truth flavour by dR matching. Without it the two
// files could not be joined -- and they cannot be joined by line order, since
// events contain different numbers of jets. The numbering is the 0-based Delphes
// entry number, which matches the generator's count of events WRITTEN to HepMC.
//
// Jet.Flavor is written by the card's JetFlavorAssociation module (|PDG| of the
// hardest parton within dR of the jet axis; 0 when none). Delphes' BTagging keys
// on this very field, so on its own it cannot validate the flavour DEFINITION --
// see the module docstring in accsim/events/btag.py, and the dR cross-check in
// analyze.py, which is the answer to that.
//
// OUTPUT (whitespace-separated, one line per jet):
//   event  pt  eta  phi  mass  flavor  btag

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "ExRootAnalysis/ExRootTreeReader.h"
#endif

#include <fstream>
#include <iomanip>
#include <iostream>

void extract_jets(const char* inFile, const char* jetsOut) {
  gSystem->Load("libDelphes");

  TChain chain("Delphes");
  chain.Add(inFile);
  ExRootTreeReader* reader = new ExRootTreeReader(&chain);
  const Long64_t nEntries = reader->GetEntries();
  TClonesArray* branchJet = reader->UseBranch("Jet");

  std::ofstream fj(jetsOut);
  fj << "# level=reco source=Delphes_Jet observable=jet_kinematics+flavor+btagmask"
     << " cols=event,pt,eta,phi,mass,flavor,btag n_entries=" << nEntries << "\n";

  long nJets = 0;
  long nB = 0, nC = 0, nLight = 0;
  for (Long64_t i = 0; i < nEntries; ++i) {
    reader->ReadEntry(i);
    for (int j = 0; j < branchJet->GetEntries(); ++j) {
      const Jet* jet = static_cast<Jet*>(branchJet->At(j));
      fj << i << " " << std::setprecision(9) << jet->PT << " " << jet->Eta << " "
         << jet->Phi << " " << jet->Mass << " " << int(jet->Flavor) << " "
         << int(jet->BTag) << "\n";
      ++nJets;
      const int f = std::abs(int(jet->Flavor));
      if (f == 5) ++nB;
      else if (f == 4) ++nC;
      else ++nLight;
    }
  }
  std::cerr << "jets: " << nJets << " in " << nEntries << " events"
            << "   (b: " << nB << ", c: " << nC << ", light/gluon: " << nLight << ")\n";
}
