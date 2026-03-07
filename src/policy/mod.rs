/// Policy distillation — zero-inference code-as-policy paths.
///
/// Implements P3 of the autopoietic agent architecture (from the AutoHarness paper's
/// "code-as-policy" variant): when the same tool invocation pattern succeeds N times,
/// distill it into a pure shell policy that executes without LLM calls.
///
/// # How It Works
///
/// 1. `PolicyDistiller` observes every tool call + result via its `observe()` method.
/// 2. It fingerprints each call: `(tool_name, arg_shape)` where arg_shape is the
///    sorted set of argument keys (not values — values vary, shapes don't).
/// 3. When the same fingerprint succeeds `distillation_threshold` times with identical
///    output patterns, it emits a `DistilledPolicy`.
/// 4. `PolicyRegistry` stores distilled policies. A `PolicyTool` wraps a policy as
///    a `Tool` — identical inputs → cached shell execution → no LLM needed.
///
/// # Cost Impact
///
/// Code-as-policy paths eliminate inference cost for known subtasks.
/// From the AutoHarness paper: code-as-policy achieved 0.870 avg reward
/// (beating GPT-5.2-High at 0.844) with zero runtime inference cost.
pub mod distiller;
pub mod registry;

pub use distiller::{CallObservation, PolicyDistiller};
pub use registry::{DistilledPolicy, PolicyRegistry, PolicyTool};
