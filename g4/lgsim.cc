// LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
// Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
// Geant4 cross-check of the light-guide response map.
//   Hollow light guide (air) with a perfectly specular wall (optical surface dielectric_metal, polished).
//   The cavity is either the ideal G4Cons or a G4TessellatedSolid built from a wall mesh r(phi_i, z_j).
//   One optical photon per event from a Lambertian point-like spot on the entrance plane; photons that
//   cross the exit plane (z = L) enter the "PMT" volume and are counted.
//
// usage: lgsim <mesh.txt | ideal> <points.txt> <photons per point> <out.txt> [spot=0] [smax=0.7] [refl=1] [seed=1]
//   spot > 0: Gaussian sigma (mm);  spot < 0: uniform disc of radius -spot (mm)
//   mesh.txt   : first line "NPHI NZ";  then NZ lines: z  r(phi_0) ... r(phi_{NPHI-1})   (mm), phi_i = 2 pi i / NPHI
//   points.txt : one "x y" (mm, light-guide centred) per line
//   out.txt    : x y  detected  emitted  T
#include "G4RunManagerFactory.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VUserPhysicsList.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4ProcessManager.hh"
#include "G4OpticalPhoton.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4ParticleGun.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iostream>

namespace {
const double R1 = 8.75 * mm, R2 = 4.0 * mm, LEN = 100.0 * mm;
std::string gMesh;
std::vector<std::pair<double, double>> gPts;
std::vector<long> gDet, gEmit;
long gPerPoint = 1000;
double gSigma = 0., gSmax = 0.7, gRefl = 1.0;
G4VSolid* gCavity = nullptr;
}

class Det : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto nist = G4NistManager::Instance();
    auto wall = nist->FindOrBuildMaterial("G4_Al");            // no RINDEX: photons can only be reflected
    auto air = new G4Material("CavityAir", 1.29e-3 * g / cm3, 1);
    air->AddMaterial(nist->FindOrBuildMaterial("G4_AIR"), 1.0);
    std::vector<G4double> en = {1.5 * eV, 4.5 * eV}, rin = {1.0, 1.0};
    auto mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", en, rin);
    air->SetMaterialPropertiesTable(mpt);

    auto worldS = new G4Box("World", 30 * mm, 30 * mm, 120 * mm);
    auto worldL = new G4LogicalVolume(worldS, wall, "World");
    auto worldP = new G4PVPlacement(nullptr, {}, worldL, "World", nullptr, false, 0);

    G4ThreeVector cavPos(0, 0, 0);
    if (gMesh == "ideal") {
      gCavity = new G4Cons("Cavity", 0, R1, 0, R2, LEN / 2, 0, twopi);   // centred at z = L/2
      cavPos = G4ThreeVector(0, 0, LEN / 2);
    } else {
      std::ifstream in(gMesh);
      int nphi, nz; in >> nphi >> nz;
      std::vector<double> z(nz); std::vector<std::vector<double>> r(nz, std::vector<double>(nphi));
      for (int j = 0; j < nz; ++j) { in >> z[j]; for (int i = 0; i < nphi; ++i) in >> r[j][i]; }
      if (!in) { G4cerr << "bad mesh file" << G4endl; exit(1); }
      auto V = [&](int i, int j) {
        double ph = twopi * (i % nphi) / nphi; double rr = r[j][i % nphi] * mm;
        return G4ThreeVector(rr * std::cos(ph), rr * std::sin(ph), z[j] * mm); };
      auto ts = new G4TessellatedSolid("Cavity");
      for (int j = 0; j + 1 < nz; ++j)
        for (int i = 0; i < nphi; ++i) {
          ts->AddFacet(new G4TriangularFacet(V(i, j), V(i + 1, j), V(i + 1, j + 1), ABSOLUTE));
          ts->AddFacet(new G4TriangularFacet(V(i, j), V(i + 1, j + 1), V(i, j + 1), ABSOLUTE));
        }
      G4ThreeVector c0(0, 0, z[0] * mm), c1(0, 0, z[nz - 1] * mm);
      for (int i = 0; i < nphi; ++i) {
        ts->AddFacet(new G4TriangularFacet(c0, V(i + 1, 0), V(i, 0), ABSOLUTE));
        ts->AddFacet(new G4TriangularFacet(c1, V(i, nz - 1), V(i + 1, nz - 1), ABSOLUTE));
      }
      ts->SetSolidClosed(true);
      gCavity = ts;
      G4cout << "tessellated cavity: " << ts->GetNumberOfFacets() << " facets, volume " << ts->GetCubicVolume() / mm3 << " mm3" << G4endl;
    }
    auto cavL = new G4LogicalVolume(gCavity, air, "Cavity");
    auto cavP = new G4PVPlacement(nullptr, cavPos, cavL, "Cavity", worldL, false, 0);
    // photon dumps on both ends (same material as the cavity -> photons walk straight in)
    auto pmtL = new G4LogicalVolume(new G4Tubs("PMT", 0, 7 * mm, 0.5 * mm, 0, twopi), air, "PMT");
    new G4PVPlacement(nullptr, {0, 0, LEN + 0.5 * mm}, pmtL, "PMT", worldL, false, 0);
    auto dumpL = new G4LogicalVolume(new G4Tubs("Dump", 0, 12 * mm, 0.5 * mm, 0, twopi), air, "Dump");
    new G4PVPlacement(nullptr, {0, 0, -0.5 * mm}, dumpL, "Dump", worldL, false, 0);

    auto surf = new G4OpticalSurface("Mirror");
    surf->SetType(dielectric_metal); surf->SetModel(unified); surf->SetFinish(polished);
    auto smpt = new G4MaterialPropertiesTable();
    std::vector<G4double> refl = {gRefl, gRefl}, eff = {0., 0.};
    smpt->AddProperty("REFLECTIVITY", en, refl);
    smpt->AddProperty("EFFICIENCY", en, eff);
    surf->SetMaterialPropertiesTable(smpt);
    new G4LogicalBorderSurface("MirrorSurface", cavP, worldP, surf);
    return worldP;
  }
};

class Gen : public G4VUserPrimaryGeneratorAction {
  G4ParticleGun gun{1};
 public:
  Gen() { gun.SetParticleDefinition(G4OpticalPhoton::Definition()); gun.SetParticleEnergy(3.0 * eV); }
  void GeneratePrimaries(G4Event* ev) override {
    size_t ip = ev->GetEventID() / gPerPoint;
    double x = gPts[ip].first * mm, y = gPts[ip].second * mm;
    if (gSigma > 0) { x += G4RandGauss::shoot(0., gSigma * mm); y += G4RandGauss::shoot(0., gSigma * mm); }
    else if (gSigma < 0) { double rr = -gSigma * mm * std::sqrt(G4UniformRand()), a = twopi * G4UniformRand(); x += rr * std::cos(a); y += rr * std::sin(a); }  // uniform disc
    double s = gSmax * std::sqrt(G4UniformRand()), ps = twopi * G4UniformRand();
    G4ThreeVector d(s * std::cos(ps), s * std::sin(ps), std::sqrt(1 - s * s));
    G4ThreeVector pos(x, y, 1e-6 * mm);
    gEmit[ip]++;
    G4ThreeVector local = pos; if (gMesh == "ideal") local -= G4ThreeVector(0, 0, LEN / 2);
    if (gCavity->Inside(local) != kInside) pos = G4ThreeVector(25 * mm, 25 * mm, 70 * mm);  // on the housing face: dies in the wall
    gun.SetParticlePosition(pos);
    gun.SetParticleMomentumDirection(d);
    G4ThreeVector pol = d.orthogonal().unit();
    gun.SetParticlePolarization(pol);
    gun.GeneratePrimaryVertex(ev);
  }
};

class Step : public G4UserSteppingAction {
 public:
  void UserSteppingAction(const G4Step* st) override {
    auto pv = st->GetPostStepPoint()->GetPhysicalVolume();
    auto trk = st->GetTrack();
    if (!pv) return;
    const G4String& n = pv->GetName();
    if (n == "PMT") {
      gDet[G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() / gPerPoint]++;
      trk->SetTrackStatus(fStopAndKill);
    } else if (n == "Dump" || trk->GetCurrentStepNumber() > 20000) {
      trk->SetTrackStatus(fStopAndKill);
    }
  }
};

// minimal physics: optical photons with transportation and the optical boundary process only
class Phys : public G4VUserPhysicsList {
 public:
  void ConstructParticle() override { G4OpticalPhoton::OpticalPhotonDefinition(); }
  void ConstructProcess() override {
    AddTransportation();
    G4OpticalPhoton::OpticalPhoton()->GetProcessManager()->AddDiscreteProcess(new G4OpBoundaryProcess());
  }
};

int main(int argc, char** argv) {
  if (argc < 5) { std::cerr << "usage: lgsim <mesh|ideal> <points> <nper> <out> [sigma] [smax] [refl] [seed]\n"; return 1; }
  gMesh = argv[1];
  { std::ifstream in(argv[2]); double x, y; while (in >> x >> y) gPts.emplace_back(x, y); }
  gPerPoint = std::atol(argv[3]);
  if (argc > 5) gSigma = std::atof(argv[5]);
  if (argc > 6) gSmax = std::atof(argv[6]);
  if (argc > 7) gRefl = std::atof(argv[7]);
  long seed = argc > 8 ? std::atol(argv[8]) : 1;
  gDet.assign(gPts.size(), 0); gEmit.assign(gPts.size(), 0);
  G4Random::setTheSeed(seed);
  auto rm = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  rm->SetUserInitialization(new Det);
  rm->SetUserInitialization(new Phys);
  rm->SetUserAction(new Gen);
  rm->SetUserAction(new Step);
  rm->SetVerboseLevel(0);
  rm->Initialize();
  rm->BeamOn(G4int(gPts.size() * gPerPoint));
  std::ofstream out(argv[4]);
  for (size_t i = 0; i < gPts.size(); ++i)
    out << gPts[i].first << " " << gPts[i].second << " " << gDet[i] << " " << gEmit[i] << " "
        << gSmax * gSmax * double(gDet[i]) / std::max(1L, gEmit[i]) << "\n";
  delete rm;
  return 0;
}
