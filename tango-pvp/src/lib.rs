pub mod battle;
pub mod eval;
pub mod game;
pub mod hooks;
pub mod input;
pub mod net;
pub mod replay;
pub mod shadow;
pub mod stepper;
pub mod sync;

// [ADDED] Register the telemetry module
pub mod telemetry; 
// Optional: Re-export for easier access
pub use telemetry::{FrameTelemetry, FrameTelemetryV1};