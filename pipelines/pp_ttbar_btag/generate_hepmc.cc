// Generate  p p -> t tbar  with Pythia8 at sqrt(s) = 13 TeV using a real LHAPDF6
// proton PDF, and write a full-event HepMC3 file (+ a small meta header).
// Compiled and run INSIDE the hepstore/rivet-pythia container (Pythia8 8.3 +
// HepMC3 + LHAPDF6). This is the generation stage of milestone E2 -- b-tagging
// performance against the Delphes card's configured working points.
//
// WHY TTBAR. The b-tagging gate needs jets of *known* flavour, and needs all
// three flavour classes the card parametrises (b / c / light+gluon) in one
// sample so that a single run measures the full efficiency-vs-mistag picture.
// ttbar delivers exactly that, which is why it is the canonical b-tagging
// calibration sample:
//   t -> W b            gives two b jets in EVERY event (the signal flavour);
//   W -> c s            gives c jets (the intermediate mistag class);
//   W -> u d / q qbar   gives light jets, and QCD radiation gives gluon jets
//                       (both fall to the card's DEFAULT formula = the mistag).
// The sibling Drell-Yan pipeline (../pp_mumu_drellyan/) forces Z -> mu mu and so
// contains no signal jets at all -- E2 could not be a re-plot of it.
//
// DECAYS ARE LEFT INCLUSIVE. Unlike the Drell-Yan chain, which forces
// 23 -> mu+ mu- to get a clean dimuon sample, nothing is forced here: the point
// is the jets, and forcing a W decay channel would bias the light/c jet mixture
// that the mistag measurement is made of. Both semi-leptonic and fully-hadronic
// ttbar are wanted.
//
// BOTH PRODUCTION CHANNELS are switched on (gg -> ttbar dominates at the LHC,
// qqbar -> ttbar is the ~10% remainder); using only one would distort the
// gluon-jet content of the sample, which is part of the mistag population.
//
// NO PHASE-SPACE MASS WINDOW is set -- ttbar production is not a resonance in
// mHat that needs a fiducial window (contrast the Drell-Yan chain, whose window
// avoids the divergent low-mass photon pole). The jet pT threshold that matters
// is applied by the Delphes card's jet finder, not here.
//
// THE PARTON DUMP, and why it exists. Delphes' BTagging module keys on exactly
// the Jet.Flavor that its JetFlavorAssociation module writes. So measuring the
// tag rate of Jet.Flavor==5 jets and recovering the card is a CLOSED LOOP: it
// validates the handling of the flavour label but cannot validate the label
// itself. To break the loop this file also dumps the generator-level heavy
// quarks, from Pythia's OWN event record (no HepMC/Delphes round-trip), so the
// host-side analysis can build an INDEPENDENT truth flavour by dR-matching jets
// to them and confirm the two labellings agree.
//
// Which b quarks: the LAST b in each chain -- a |PID|==5 particle with no
// |PID|==5 daughter, i.e. the one that goes on to hadronise. Selecting on status
// codes was deliberately avoided: this project already found that Pythia status
// codes do not survive the HepMC3 round-trip (see ../ee_mumu_delphes/README.md),
// and while this dump is pre-HepMC, keeping the definition status-free makes it
// robust and means the same rule can be applied on either side.
//
// Build: g++ generate_hepmc.cc -o gen $(pythia8-config --cxxflags --libs) -lHepMC3

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

#include <cmath>
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

// True when `p` carries flavour `pid` but none of its daughters do -- the last
// heavy quark of its chain, the one that hadronises. Pythia's daughter1/daughter2
// are a RANGE, not two indices, so the whole span is scanned.
static bool isLastOfFlavour(const Event& ev, int i, int pid) {
  const Particle& p = ev[i];
  if (std::abs(p.id()) != pid) return false;
  const int d1 = p.daughter1(), d2 = p.daughter2();
  if (d1 == 0 && d2 == 0) return true;  // no daughters recorded
  const int lo = std::min(d1, d2), hi = std::max(d1, d2);
  for (int d = lo; d <= hi; ++d) {
    if (d <= 0 || d >= ev.size()) continue;
    if (std::abs(ev[d].id()) == pid) return false;
  }
  return true;
}

int main() {
  const double sqrtS = env_d("TT_SQRT_S", 13000.0);
  const int nEvents = env_i("TT_N", 20000);
  const int seed = env_i("TT_SEED", 20260720);
  const char* pdfSet = env_s("TT_PDF_SET", "NNPDF31_lo_as_0118");
  const char* metaPath = env_s("TT_META", "/tmp/meta.dat");
  const char* hepmcPath = env_s("TT_HEPMC", "/tmp/events.hepmc");
  const char* partonPath = env_s("TT_PARTONS", "/tmp/gen_partons.dat");
  // Only dump partons hard enough to seed a jet the card's finder will keep.
  const double partonPtMin = env_d("TT_PARTON_PT_MIN", 5.0);

  Pythia pythia;
  pythia.readString("Beams:idA = 2212");  // proton
  pythia.readString("Beams:idB = 2212");  // proton
  pythia.settings.parm("Beams:eCM", sqrtS);

  // Real LHAPDF6 proton PDF, member 0. An LO set to match Pythia's LO matrix
  // element. If the set is missing, init fails -- the driver runs `lhapdf get`
  // first and reports a clean error rather than a cryptic Pythia crash.
  pythia.settings.word("PDF:pSet", std::string("LHAPDF6:") + pdfSet + "/0");

  // ttbar production, both channels.
  pythia.readString("Top:gg2ttbar = on");
  pythia.readString("Top:qqbar2ttbar = on");

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

  // Generator-level heavy-quark dump, for the INDEPENDENT truth flavour label.
  // One line per quark: the event index ties it back to the jets, which are
  // written by the Delphes macro with the same 0-based event numbering.
  //   event  pid  pt  eta  phi
  std::ofstream fp(partonPath);
  fp << "# level=gen_parton source=Pythia selection=last_of_chain"
     << " cols=event,pid,pt,eta,phi pt_min_GeV=" << partonPtMin << "\n";

  long nB = 0, nC = 0, nWritten = 0;
  for (int iE = 0; iE < nEvents; ++iE) {
    if (!pythia.next()) continue;
    toHepMC.writeNextEvent(pythia);  // Delphes sees the full event (incl. ISR/FSR).

    for (int i = 0; i < pythia.event.size(); ++i) {
      const Particle& p = pythia.event[i];
      const int a = std::abs(p.id());
      if (a != 4 && a != 5) continue;
      if (!isLastOfFlavour(pythia.event, i, a)) continue;
      if (p.pT() < partonPtMin) continue;
      // The event index MUST be the HepMC entry number, i.e. a count of events
      // actually WRITTEN -- not the loop counter iE, which also advances on the
      // pythia.next() failures that write nothing. Getting this wrong would
      // silently misalign every jet with another event's partons.
      fp << std::setprecision(9) << nWritten << " " << p.id() << " " << p.pT() << " "
         << p.eta() << " " << p.phi() << "\n";
      if (a == 5) ++nB;
      else ++nC;
    }
    ++nWritten;
  }

  const double sigma_mb = pythia.info.sigmaGen();
  const double sigma_err_mb = pythia.info.sigmaErr();
  const double version = pythia.parm("Pythia:versionNumber");

  std::ofstream fh(metaPath);
  fh << "# process=pp->ttbar sqrt_s_GeV=" << sqrtS << " n_generated=" << nEvents
     << " n_accepted=" << nWritten << " pdf_set=" << pdfSet << " pdf_member=0"
     << " sigma_mb=" << sigma_mb << " sigma_err_mb=" << sigma_err_mb
     << " n_gen_b=" << nB << " n_gen_c=" << nC
     << " parton_pt_min_GeV=" << partonPtMin
     << " pythia_version=" << version << "\n";

  std::cerr << "wrote HepMC3 to " << hepmcPath << " (" << nWritten << " events); PDF="
            << pdfSet << "; Pythia sigma = " << sigma_mb * 1e9 << " +/- "
            << sigma_err_mb * 1e9 << " pb (inclusive ttbar)\n";
  std::cerr << "wrote " << nB << " gen b and " << nC << " gen c quarks to "
            << partonPath << "\n";
  return 0;
}
