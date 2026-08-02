//! The formal path, built beside the legacy tessellator.
//!
//! `MATHEMATICAL_FOUNDATION.md` §0 states the design rule this subtree exists
//! to obey:
//!
//! > Every pipeline stage is a fallible constructor. Its output type
//! > represents a stronger state and carries evidence for the obligations that
//! > were discharged.
//!
//! **Step 1 only.** This subtree currently contains the ambient-period
//! authority model, the intermediate judgment algebra, and the bounded
//! evaluation policy. There is no traversal classification, no projection, no
//! source-local lifting, no deck solving, no finite-cover enumeration, no
//! arrangement, no material selection and no meshing. `FORMAL_SYSTEM.md`
//! §XVIII lists those stages; each arrives with its own step.
//!
//! **Nothing here is read by production geometry.** `triangulation.rs` gets a
//! module declaration and one diagnostic probe. The legacy producer-consumer
//! chain is untouched, and the measured Step 0 baseline — 1,358,543 triangles,
//! 4,486 faces lost — is unchanged by construction, because no value computed
//! in this subtree reaches it.
//!
//! # The layout
//!
//! - [`numeric`] — checked finite wrappers. No formal certificate holds a raw
//!   `f64` bound.
//! - [`evidence`] — `Fact(P, κ)` of Definition 2, as a sum type whose
//!   `Unresolved` state carries no value, with proposition-specific
//!   admissibility policies rather than a confidence ranking.
//! - [`outcome`] — the four result layers: stage result, final semantic
//!   outcome, realization outcome, operational failure.
//! - [`envelope`] — the bounds `β` of Definition 6 and the separate execution
//!   budget.
//! - [`ambient`] — `Λ` of Definition 7: the five distinguishable states of an
//!   axis's period evidence, and the rank-structured lattice they can resolve
//!   to.
//!
//! # What the subtree will not let you do
//!
//! Authority is minted only by named introduction rules. `Evidence`'s three
//! authority-bearing constructors, `PeriodCertificate::new`,
//! `CertifiedPeriodGenerator::new`, `ExactCount::from_completed_count`,
//! `CertifiedLowerBoundCount::from_certificate` and
//! `FeatureExclusionWitness::new` are all `pub(super)` and are *not*
//! re-exported below. Outside this subtree the only way to obtain a certified
//! ambient generator is [`ambient::certify_revolution_period`] or
//! [`ambient::certify_period_numerically`], and the only way to obtain a
//! certified lattice is [`ambient::resolve_ambient_periods`].

pub mod ambient;
pub mod envelope;
pub mod evidence;
pub mod numeric;
pub mod outcome;

// The Step 1 surface, re-exported for the probe and for later stages. Types
// only: every proof-introduction rule stays behind its module's visibility.
pub use ambient::{
    ambient_axis_evidence_from_legacy, ambient_evidence_from_legacy, certify_period_numerically,
    certify_plane_aperiodicity, certify_revolution_period,
    certify_straight_generatrix_aperiodicity, resolve_ambient_periods, AdapterError,
    AmbientAlternative, AmbientPeriodEvidence, CertifiedAmbientLattice, CertifiedPeriodBasisRef,
    CertifiedPeriodGenerator, CertifiedRank0, CertifiedRank1, CertifiedRank2,
    CertifiedUvTranslation, CoverEnumerationAuthority, DeckDisplacement, DeckVector0, DeckVector1,
    DeckVector2, DeclaredPeriod, DeclaredPeriodHint, GeneratorIndependenceCertificate,
    LatticeOrigin, ObservedPeriod, PeriodAbsence, PeriodAxisEvidence, PeriodCertificate,
    PeriodCertificationAttempt, PeriodCertificationFailure, PeriodContradictionWitness,
    PeriodHintSet, PeriodHintSource, QuotientIdentificationAuthority, UncertifiedPeriodValue,
};
pub use envelope::{
    BoundObservation, CertifiedLowerBoundCount, ExactCount, ExecutionBudget, FeatureExclusion,
    FeatureExclusionWitness, FormalEnvelope, NumericEnvelopeClause, PolicyInstanceId,
};
pub use evidence::{
    AnalyticCertificate, AnalyticPremise, AnalyticRule, AtLeastTwo, AuthoritativeBasis,
    AuthoritativeFact, Evidence, EvidenceCertificate, EvidenceRequirement, EvidenceStatus,
    FormalPredicate, NonEmptyVec, NumericalCertificate, ParameterAxis, PredicateDescription,
    SemanticStage, SourceEntityKey, ANALYTIC_ONLY, ANALYTIC_OR_CERTIFIED_NUMERICAL,
    ANY_AUTHORITATIVE, SOURCE_DECLARATION,
};
pub use outcome::{
    Ambiguity, AmbiguityReport, ContradictionWitness, DocumentKey, DocumentScope, FaceKey,
    Inconsistency, InconsistencyReport, OperationalFailure, RealizationOutcome, SemanticOutcome,
    ShellKey, StageEvaluation, StageOutcome, UnresolvedReason, UnresolvedReport, UnsupportedCause,
    UnsupportedReport, ValidSemantic,
};
