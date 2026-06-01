//! # lau-persistence-experiment
//!
//! Tests the hypothesis that the persistence diagram of an agent's belief manifold
//! predicts its learning trajectory.
//!
//! **Core hypothesis:** Agents whose belief manifolds have long-lived topological
//! features (high persistence) learn more slowly but more robustly. Agents with
//! short-lived features learn fast but are fragile.

mod belief;
mod persistence;
mod betti;
mod landscape;
mod complexity;
mod birth_death;
mod phase;
mod robustness;
mod comparison;
mod falsification;

pub use belief::*;
pub use persistence::*;
pub use betti::*;
pub use landscape::*;
pub use complexity::*;
pub use birth_death::*;
pub use phase::*;
pub use robustness::*;
pub use comparison::*;
pub use falsification::*;
