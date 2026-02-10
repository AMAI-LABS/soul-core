//! Planner — task graph with dependencies, status tracking, and display formatting.
//!
//! A directed acyclic graph of tasks where each task can block or be blocked by
//! other tasks. Supports status progression (pending → in_progress → completed),
//! timing, and visual rendering similar to Claude Code's task display.
//!
//! ```rust
//! use soul_core::planner::{Planner, TaskStatus};
//!
//! let mut planner = Planner::new();
//! let t1 = planner.add_task("Implement VFS trait", Some("Implementing VFS trait"));
//! let t2 = planner.add_task("Write tests", Some("Writing tests"));
//! let t3 = planner.add_task("Deploy", Some("Deploying"));
//! planner.add_dependency(t3, t1); // t3 blocked by t1
//! planner.add_dependency(t3, t2); // t3 blocked by t1, t2
//!
//! planner.start(t1);
//! planner.complete(t1);
//! planner.start(t2);
//! planner.complete(t2);
//! // t3 is now unblocked
//! assert!(planner.is_ready(t3));
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{SoulError, SoulResult};

/// Unique task identifier (monotonically increasing).
pub type TaskId = u64;

/// Unique question identifier (monotonically increasing).
pub type QuestionId = u64;

/// Task status progression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl TaskStatus {
    /// Symbol for display rendering.
    pub fn symbol(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "◻",
            TaskStatus::InProgress => "◼",
            TaskStatus::Completed => "✓",
            TaskStatus::Failed => "✗",
            TaskStatus::Skipped => "⊘",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Skipped
        )
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// A single task in the planner graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanTask {
    pub id: TaskId,
    pub subject: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    /// Present-continuous form for spinner display (e.g., "Implementing VFS trait")
    pub active_form: Option<String>,
    /// Tasks that must complete before this one can start.
    pub blocked_by: BTreeSet<TaskId>,
    /// Tasks that this task blocks (inverse of blocked_by).
    pub blocks: BTreeSet<TaskId>,
    /// Optional metadata.
    pub metadata: HashMap<String, String>,
    /// Elapsed duration (accumulated across start/stop cycles).
    #[serde(default)]
    pub elapsed_ms: u64,
    /// Creation order for stable display.
    #[serde(default)]
    pub order: u64,
    /// Current attempt number (1-indexed, incremented on retry).
    #[serde(default = "default_attempt")]
    pub attempt: u32,
    /// Max retries before permanent failure (0 = no retry).
    #[serde(default)]
    pub max_retries: u32,
    /// Error from the last failed attempt.
    #[serde(default)]
    pub last_error: Option<String>,
    /// Checkpoint: partial or final output from execution.
    #[serde(default)]
    pub output: Option<String>,
}

fn default_attempt() -> u32 {
    1
}

impl PlanTask {
    fn new(id: TaskId, subject: impl Into<String>, active_form: Option<String>) -> Self {
        Self {
            id,
            subject: subject.into(),
            description: None,
            status: TaskStatus::Pending,
            active_form,
            blocked_by: BTreeSet::new(),
            blocks: BTreeSet::new(),
            metadata: HashMap::new(),
            elapsed_ms: 0,
            order: id,
            attempt: 1,
            max_retries: 0,
            last_error: None,
            output: None,
        }
    }

    /// Whether this task can be retried (has retries left).
    pub fn can_retry(&self) -> bool {
        self.status == TaskStatus::Failed && self.attempt <= self.max_retries
    }

    /// Whether all blockers are in terminal state.
    pub fn is_unblocked(&self, tasks: &BTreeMap<TaskId, PlanTask>) -> bool {
        self.blocked_by.iter().all(|dep_id| {
            tasks
                .get(dep_id)
                .map(|t| t.status.is_terminal())
                .unwrap_or(true) // missing task = unblocked
        })
    }

    /// Format the blocked-by annotation.
    pub fn blocked_by_annotation(&self) -> Option<String> {
        if self.blocked_by.is_empty() {
            return None;
        }
        let ids: Vec<String> = self.blocked_by.iter().map(|id| format!("#{id}")).collect();
        Some(format!("blocked by {}", ids.join(", ")))
    }
}

/// A clarifying question for the user — last resort for complex plans.
///
/// Questions are stored in the planner and surfaced to the consuming code
/// (agent loop, CLI, etc.) which handles the actual IO. The planner itself
/// never performs IO.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanQuestion {
    pub id: QuestionId,
    pub question: String,
    /// Optional choices (if empty, freeform answer expected).
    #[serde(default)]
    pub options: Vec<String>,
    /// The user's answer, if provided.
    pub answer: Option<String>,
    /// Which task(s) this question relates to, if any.
    #[serde(default)]
    pub related_tasks: BTreeSet<TaskId>,
}

impl PlanQuestion {
    fn new(id: QuestionId, question: impl Into<String>) -> Self {
        Self {
            id,
            question: question.into(),
            options: Vec::new(),
            answer: None,
            related_tasks: BTreeSet::new(),
        }
    }

    /// Whether this question has been answered.
    pub fn is_answered(&self) -> bool {
        self.answer.is_some()
    }
}

/// Serializable snapshot of a Planner (no Instant fields).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlannerSnapshot {
    pub tasks: BTreeMap<TaskId, PlanTask>,
    pub next_id: TaskId,
    #[serde(default)]
    pub questions: BTreeMap<QuestionId, PlanQuestion>,
    #[serde(default)]
    pub next_question_id: QuestionId,
}

impl PlannerSnapshot {
    pub fn to_json(&self) -> SoulResult<String> {
        serde_json::to_string(self).map_err(SoulError::Serialization)
    }

    pub fn to_json_pretty(&self) -> SoulResult<String> {
        serde_json::to_string_pretty(self).map_err(SoulError::Serialization)
    }

    pub fn from_json(json: &str) -> SoulResult<Self> {
        serde_json::from_str(json).map_err(SoulError::Serialization)
    }
}

/// Task graph with dependency tracking, timing, and display formatting.
pub struct Planner {
    tasks: BTreeMap<TaskId, PlanTask>,
    next_id: TaskId,
    /// Runtime-only: tracks when an in-progress task was started (not serialized).
    start_times: HashMap<TaskId, Instant>,
    /// Clarifying questions for the user (last resort for complex plans).
    questions: BTreeMap<QuestionId, PlanQuestion>,
    next_question_id: QuestionId,
}

impl Planner {
    pub fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
            next_id: 1,
            start_times: HashMap::new(),
            questions: BTreeMap::new(),
            next_question_id: 1,
        }
    }

    /// Restore from a serialized snapshot.
    pub fn from_snapshot(snapshot: PlannerSnapshot) -> Self {
        Self {
            tasks: snapshot.tasks,
            next_id: snapshot.next_id,
            start_times: HashMap::new(),
            questions: snapshot.questions,
            next_question_id: snapshot.next_question_id,
        }
    }

    /// Restore from JSON.
    pub fn from_json(json: &str) -> SoulResult<Self> {
        let snapshot = PlannerSnapshot::from_json(json)?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Capture current state as a serializable snapshot.
    pub fn snapshot(&self) -> PlannerSnapshot {
        // Flush any running timers into elapsed_ms before snapshotting
        let mut tasks = self.tasks.clone();
        let now = Instant::now();
        for (id, start) in &self.start_times {
            if let Some(task) = tasks.get_mut(id) {
                task.elapsed_ms += now.duration_since(*start).as_millis() as u64;
            }
        }
        PlannerSnapshot {
            tasks,
            next_id: self.next_id,
            questions: self.questions.clone(),
            next_question_id: self.next_question_id,
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> SoulResult<String> {
        self.snapshot().to_json()
    }

    /// Add a new task. Returns its ID.
    pub fn add_task(
        &mut self,
        subject: impl Into<String>,
        active_form: Option<impl Into<String>>,
    ) -> TaskId {
        let id = self.next_id;
        self.next_id += 1;
        let task = PlanTask::new(id, subject, active_form.map(|s| s.into()));
        self.tasks.insert(id, task);
        id
    }

    /// Add a task with description.
    pub fn add_task_with_description(
        &mut self,
        subject: impl Into<String>,
        description: impl Into<String>,
        active_form: Option<impl Into<String>>,
    ) -> TaskId {
        let id = self.add_task(subject, active_form);
        if let Some(task) = self.tasks.get_mut(&id) {
            task.description = Some(description.into());
        }
        id
    }

    /// Add a dependency: `task_id` is blocked by `blocked_by_id`.
    pub fn add_dependency(&mut self, task_id: TaskId, blocked_by_id: TaskId) -> SoulResult<()> {
        // Check both tasks exist
        if !self.tasks.contains_key(&task_id) {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Task #{task_id} not found"
            )));
        }
        if !self.tasks.contains_key(&blocked_by_id) {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Task #{blocked_by_id} not found"
            )));
        }
        // Prevent self-dependency
        if task_id == blocked_by_id {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Task #{task_id} cannot depend on itself"
            )));
        }
        // Check for cycles: would adding this edge create a cycle?
        if self.would_create_cycle(task_id, blocked_by_id) {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Dependency #{task_id} → #{blocked_by_id} would create a cycle"
            )));
        }

        self.tasks
            .get_mut(&task_id)
            .unwrap()
            .blocked_by
            .insert(blocked_by_id);
        self.tasks
            .get_mut(&blocked_by_id)
            .unwrap()
            .blocks
            .insert(task_id);
        Ok(())
    }

    /// Check if adding task_id blocked_by blocked_by_id would create a cycle.
    /// A cycle exists if blocked_by_id is reachable from task_id through the "blocks" graph.
    fn would_create_cycle(&self, task_id: TaskId, blocked_by_id: TaskId) -> bool {
        // DFS from blocked_by_id following blocked_by edges — if we reach task_id, cycle.
        let mut stack = vec![blocked_by_id];
        let mut visited = BTreeSet::new();
        while let Some(current) = stack.pop() {
            if current == task_id {
                return true;
            }
            if visited.insert(current) {
                if let Some(task) = self.tasks.get(&current) {
                    for &dep in &task.blocked_by {
                        stack.push(dep);
                    }
                }
            }
        }
        false
    }

    /// Start a task (set to InProgress), recording the start time.
    pub fn start(&mut self, id: TaskId) -> SoulResult<()> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| SoulError::Other(anyhow::anyhow!("Task #{id} not found")))?;

        if task.status.is_terminal() {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Task #{id} is already in terminal state: {:?}",
                task.status
            )));
        }
        task.status = TaskStatus::InProgress;
        self.start_times.insert(id, Instant::now());
        Ok(())
    }

    /// Complete a task, recording elapsed time.
    pub fn complete(&mut self, id: TaskId) -> SoulResult<()> {
        self.finish_task(id, TaskStatus::Completed)
    }

    /// Fail a task.
    pub fn fail(&mut self, id: TaskId) -> SoulResult<()> {
        self.finish_task(id, TaskStatus::Failed)
    }

    /// Skip a task.
    pub fn skip(&mut self, id: TaskId) -> SoulResult<()> {
        self.finish_task(id, TaskStatus::Skipped)
    }

    fn finish_task(&mut self, id: TaskId, status: TaskStatus) -> SoulResult<()> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| SoulError::Other(anyhow::anyhow!("Task #{id} not found")))?;
        // Record elapsed time if we were tracking
        if let Some(start) = self.start_times.remove(&id) {
            task.elapsed_ms += start.elapsed().as_millis() as u64;
        }
        task.status = status;
        Ok(())
    }

    /// Fail a task with an error message.
    pub fn fail_with_error(&mut self, id: TaskId, error: impl Into<String>) -> SoulResult<()> {
        let error_msg = error.into();
        self.finish_task(id, TaskStatus::Failed)?;
        if let Some(task) = self.tasks.get_mut(&id) {
            task.last_error = Some(error_msg);
        }
        Ok(())
    }

    /// Retry a failed task — reset to pending, increment attempt counter.
    ///
    /// Returns `Err` if the task doesn't exist, isn't failed, or has exhausted retries.
    pub fn retry(&mut self, id: TaskId) -> SoulResult<()> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| SoulError::Other(anyhow::anyhow!("Task #{id} not found")))?;

        if task.status != TaskStatus::Failed {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Task #{id} is {:?}, can only retry failed tasks",
                task.status
            )));
        }
        if task.attempt > task.max_retries {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Task #{id} exhausted retries ({}/{})",
                task.attempt,
                task.max_retries
            )));
        }

        task.attempt += 1;
        task.status = TaskStatus::Pending;
        // Keep last_error and output for context — don't clear them
        Ok(())
    }

    /// Auto-retry all failed tasks that have retries remaining.
    /// Returns the IDs of tasks that were retried.
    pub fn retry_all(&mut self) -> Vec<TaskId> {
        let retryable: Vec<TaskId> = self
            .tasks
            .values()
            .filter(|t| t.can_retry())
            .map(|t| t.id)
            .collect();

        let mut retried = Vec::new();
        for id in retryable {
            if self.retry(id).is_ok() {
                retried.push(id);
            }
        }
        retried
    }

    /// Resume after interruption (crash, laptop close, process kill).
    ///
    /// All `InProgress` tasks are reset to `Pending` so they can be re-executed.
    /// Elapsed time is preserved. Returns the IDs of tasks that were reset.
    pub fn resume(&mut self) -> Vec<TaskId> {
        let in_flight: Vec<TaskId> = self
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::InProgress)
            .map(|t| t.id)
            .collect();

        for &id in &in_flight {
            if let Some(task) = self.tasks.get_mut(&id) {
                task.status = TaskStatus::Pending;
                task.attempt += 1;
                // Flush any live timer into elapsed_ms
                if let Some(start) = self.start_times.remove(&id) {
                    task.elapsed_ms += start.elapsed().as_millis() as u64;
                }
            }
        }
        in_flight
    }

    /// Store a checkpoint (partial or final output) on a task.
    ///
    /// Survives serialization — if the process dies, the last checkpoint
    /// is available after `from_snapshot()` + `resume()`.
    pub fn checkpoint(&mut self, id: TaskId, output: impl Into<String>) -> SoulResult<()> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| SoulError::Other(anyhow::anyhow!("Task #{id} not found")))?;
        task.output = Some(output.into());
        Ok(())
    }

    /// Set the max retry count for a task.
    pub fn set_max_retries(&mut self, id: TaskId, max_retries: u32) -> SoulResult<()> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| SoulError::Other(anyhow::anyhow!("Task #{id} not found")))?;
        task.max_retries = max_retries;
        Ok(())
    }

    /// Check if a task is ready to start (all dependencies completed).
    pub fn is_ready(&self, id: TaskId) -> bool {
        self.tasks
            .get(&id)
            .map(|task| task.status == TaskStatus::Pending && task.is_unblocked(&self.tasks))
            .unwrap_or(false)
    }

    /// Get a task by ID.
    pub fn get(&self, id: TaskId) -> Option<&PlanTask> {
        self.tasks.get(&id)
    }

    /// Get a mutable reference to a task.
    pub fn get_mut(&mut self, id: TaskId) -> Option<&mut PlanTask> {
        self.tasks.get_mut(&id)
    }

    /// Remove a task and clean up all dependency edges.
    pub fn remove(&mut self, id: TaskId) -> Option<PlanTask> {
        let task = self.tasks.remove(&id)?;
        self.start_times.remove(&id);
        // Remove from other tasks' blocked_by / blocks
        for &blocked in &task.blocks {
            if let Some(t) = self.tasks.get_mut(&blocked) {
                t.blocked_by.remove(&id);
            }
        }
        for &dep in &task.blocked_by {
            if let Some(t) = self.tasks.get_mut(&dep) {
                t.blocks.remove(&id);
            }
        }
        Some(task)
    }

    /// All tasks in creation order.
    pub fn all_tasks(&self) -> Vec<&PlanTask> {
        let mut tasks: Vec<&PlanTask> = self.tasks.values().collect();
        tasks.sort_by_key(|t| t.order);
        tasks
    }

    /// Tasks that are ready to start (pending + unblocked).
    pub fn ready_tasks(&self) -> Vec<&PlanTask> {
        self.tasks
            .values()
            .filter(|t| t.status == TaskStatus::Pending && t.is_unblocked(&self.tasks))
            .collect()
    }

    /// Tasks currently in progress.
    pub fn in_progress_tasks(&self) -> Vec<&PlanTask> {
        self.tasks
            .values()
            .filter(|t| t.status == TaskStatus::InProgress)
            .collect()
    }

    /// Tasks that are blocked (pending but have incomplete dependencies).
    pub fn blocked_tasks(&self) -> Vec<&PlanTask> {
        self.tasks
            .values()
            .filter(|t| t.status == TaskStatus::Pending && !t.is_unblocked(&self.tasks))
            .collect()
    }

    /// Next task to work on: first ready task by creation order.
    pub fn next_task(&self) -> Option<&PlanTask> {
        let mut ready = self.ready_tasks();
        ready.sort_by_key(|t| t.order);
        ready.into_iter().next()
    }

    /// Count tasks by status.
    pub fn counts(&self) -> PlannerCounts {
        let mut counts = PlannerCounts::default();
        for task in self.tasks.values() {
            counts.total += 1;
            match task.status {
                TaskStatus::Pending => counts.pending += 1,
                TaskStatus::InProgress => counts.in_progress += 1,
                TaskStatus::Completed => counts.completed += 1,
                TaskStatus::Failed => counts.failed += 1,
                TaskStatus::Skipped => counts.skipped += 1,
            }
        }
        counts
    }

    /// Whether all tasks are in a terminal state.
    pub fn is_done(&self) -> bool {
        self.tasks.values().all(|t| t.status.is_terminal())
    }

    /// Total number of tasks.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Format the current elapsed time for an in-progress task.
    pub fn elapsed(&self, id: TaskId) -> Duration {
        let base_ms = self.tasks.get(&id).map(|t| t.elapsed_ms).unwrap_or(0);
        let live_ms = self
            .start_times
            .get(&id)
            .map(|s| s.elapsed().as_millis() as u64)
            .unwrap_or(0);
        Duration::from_millis(base_ms + live_ms)
    }

    // ─── Clarifying Questions ────────────────────────────────────────────

    /// Add a clarifying question. Returns its ID.
    ///
    /// Questions are last resort — only for genuinely ambiguous requirements
    /// in complex plans. The consuming code handles the actual user IO.
    pub fn add_question(&mut self, question: impl Into<String>) -> QuestionId {
        let id = self.next_question_id;
        self.next_question_id += 1;
        self.questions.insert(id, PlanQuestion::new(id, question));
        id
    }

    /// Add a question with multiple-choice options.
    pub fn add_question_with_options(
        &mut self,
        question: impl Into<String>,
        options: Vec<String>,
    ) -> QuestionId {
        let id = self.add_question(question);
        if let Some(q) = self.questions.get_mut(&id) {
            q.options = options;
        }
        id
    }

    /// Link a question to specific task(s).
    pub fn link_question_to_task(
        &mut self,
        question_id: QuestionId,
        task_id: TaskId,
    ) -> SoulResult<()> {
        let q = self.questions.get_mut(&question_id).ok_or_else(|| {
            SoulError::Other(anyhow::anyhow!("Question #{question_id} not found"))
        })?;
        q.related_tasks.insert(task_id);
        Ok(())
    }

    /// Record an answer to a question.
    pub fn answer_question(
        &mut self,
        question_id: QuestionId,
        answer: impl Into<String>,
    ) -> SoulResult<()> {
        let q = self.questions.get_mut(&question_id).ok_or_else(|| {
            SoulError::Other(anyhow::anyhow!("Question #{question_id} not found"))
        })?;
        q.answer = Some(answer.into());
        Ok(())
    }

    /// Get a question by ID.
    pub fn get_question(&self, id: QuestionId) -> Option<&PlanQuestion> {
        self.questions.get(&id)
    }

    /// All questions.
    pub fn all_questions(&self) -> Vec<&PlanQuestion> {
        self.questions.values().collect()
    }

    /// Questions that haven't been answered yet.
    pub fn open_questions(&self) -> Vec<&PlanQuestion> {
        self.questions
            .values()
            .filter(|q| !q.is_answered())
            .collect()
    }

    /// Whether there are unanswered questions.
    pub fn has_open_questions(&self) -> bool {
        self.questions.values().any(|q| !q.is_answered())
    }

    /// Whether the plan is ready to execute (no open questions).
    pub fn is_ready_to_execute(&self) -> bool {
        !self.has_open_questions()
    }

    /// Render the plan as a display string, similar to Claude Code's task rendering.
    ///
    /// Format:
    /// ```text
    /// ✽ Implementing VFS trait… (7m 24s)
    /// ⎿ ◼ Implement VFS trait
    ///   ✓ Write core types
    ///   ◻ Deploy › blocked by #1, #2
    /// ```
    pub fn render(&self) -> String {
        let mut lines = Vec::new();

        // Find the active task for the header
        let active = self.in_progress_tasks();
        if let Some(task) = active.first() {
            let label = task.active_form.as_deref().unwrap_or(&task.subject);
            let elapsed = self.elapsed(task.id);
            let time_str = format_duration(elapsed);
            lines.push(format!("✽ {label}… ({time_str})"));
        }

        // Render all tasks
        let all = self.all_tasks();
        for (i, task) in all.iter().enumerate() {
            let prefix = if !active.is_empty() && i == 0 {
                "⎿ "
            } else if !active.is_empty() {
                "  "
            } else {
                ""
            };

            let symbol = task.status.symbol();
            let mut line = format!("{prefix}{symbol} {}", task.subject);

            // Add blocked-by annotation
            if let Some(annotation) = task.blocked_by_annotation() {
                if !task.status.is_terminal() {
                    line.push_str(&format!(" › {annotation}"));
                }
            }

            // Add retry annotation for failed tasks
            if task.status == TaskStatus::Failed && task.max_retries > 0 {
                line.push_str(&format!(
                    " (attempt {}/{})",
                    task.attempt,
                    task.max_retries + 1
                ));
            }

            // Add elapsed time for completed tasks
            if task.status == TaskStatus::Completed && task.elapsed_ms > 0 {
                let time_str = format_duration(Duration::from_millis(task.elapsed_ms));
                line.push_str(&format!(" ({time_str})"));
            }

            lines.push(line);
        }

        // Summary line
        let counts = self.counts();
        if counts.total > 0 {
            lines.push(format!("\n{}/{} completed", counts.completed, counts.total));
        }

        // Open questions
        let open = self.open_questions();
        if !open.is_empty() {
            lines.push(String::new());
            lines.push(format!("? {} open question(s):", open.len()));
            for q in &open {
                let mut line = format!("  ? {}", q.question);
                if !q.options.is_empty() {
                    let opts = q.options.join(" / ");
                    line.push_str(&format!(" [{opts}]"));
                }
                lines.push(line);
            }
        }

        lines.join("\n")
    }

    /// Topological sort — returns task IDs in dependency-respecting order.
    pub fn topological_order(&self) -> SoulResult<Vec<TaskId>> {
        let mut in_degree: HashMap<TaskId, usize> = HashMap::new();
        for task in self.tasks.values() {
            in_degree.entry(task.id).or_insert(0);
            for &blocked in &task.blocks {
                *in_degree.entry(blocked).or_insert(0) += 1;
            }
        }

        let mut queue: Vec<TaskId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort(); // deterministic order

        let mut result = Vec::new();
        while let Some(id) = queue.pop() {
            result.push(id);
            if let Some(task) = self.tasks.get(&id) {
                for &blocked in &task.blocks {
                    if let Some(deg) = in_degree.get_mut(&blocked) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(blocked);
                            queue.sort();
                        }
                    }
                }
            }
        }

        if result.len() != self.tasks.len() {
            return Err(SoulError::Other(anyhow::anyhow!(
                "Cycle detected in task graph"
            )));
        }
        Ok(result)
    }
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

/// Task counts by status.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannerCounts {
    pub total: usize,
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub failed: usize,
    pub skipped: usize,
}

/// Format a Duration as human-readable (e.g., "7m 24s", "1h 2m", "45s").
pub fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Basic Task Operations ──────────────────────────────────────────

    #[test]
    fn add_and_get_task() {
        let mut planner = Planner::new();
        let id = planner.add_task("Build feature", Some("Building feature"));
        assert_eq!(id, 1);

        let task = planner.get(id).unwrap();
        assert_eq!(task.subject, "Build feature");
        assert_eq!(task.active_form.as_deref(), Some("Building feature"));
        assert_eq!(task.status, TaskStatus::Pending);
    }

    #[test]
    fn add_task_with_description() {
        let mut planner = Planner::new();
        let id = planner.add_task_with_description(
            "Complex task",
            "This is a detailed description of the task",
            Some("Working on complex task"),
        );
        let task = planner.get(id).unwrap();
        assert_eq!(
            task.description.as_deref(),
            Some("This is a detailed description of the task")
        );
    }

    #[test]
    fn auto_incrementing_ids() {
        let mut planner = Planner::new();
        let a = planner.add_task("A", None::<String>);
        let b = planner.add_task("B", None::<String>);
        let c = planner.add_task("C", None::<String>);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    }

    #[test]
    fn task_count() {
        let mut planner = Planner::new();
        assert!(planner.is_empty());
        assert_eq!(planner.len(), 0);
        planner.add_task("A", None::<String>);
        planner.add_task("B", None::<String>);
        assert_eq!(planner.len(), 2);
        assert!(!planner.is_empty());
    }

    // ─── Status Transitions ─────────────────────────────────────────────

    #[test]
    fn status_lifecycle() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        assert_eq!(planner.get(id).unwrap().status, TaskStatus::Pending);

        planner.start(id).unwrap();
        assert_eq!(planner.get(id).unwrap().status, TaskStatus::InProgress);

        planner.complete(id).unwrap();
        assert_eq!(planner.get(id).unwrap().status, TaskStatus::Completed);
    }

    #[test]
    fn fail_task() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        planner.start(id).unwrap();
        planner.fail(id).unwrap();
        assert_eq!(planner.get(id).unwrap().status, TaskStatus::Failed);
    }

    #[test]
    fn skip_task() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        planner.skip(id).unwrap();
        assert_eq!(planner.get(id).unwrap().status, TaskStatus::Skipped);
    }

    #[test]
    fn cannot_start_completed_task() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        planner.start(id).unwrap();
        planner.complete(id).unwrap();
        assert!(planner.start(id).is_err());
    }

    #[test]
    fn cannot_start_nonexistent_task() {
        let mut planner = Planner::new();
        assert!(planner.start(999).is_err());
    }

    // ─── Dependencies ───────────────────────────────────────────────────

    #[test]
    fn add_dependency() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("First", None::<String>);
        let t2 = planner.add_task("Second", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        let task2 = planner.get(t2).unwrap();
        assert!(task2.blocked_by.contains(&t1));

        let task1 = planner.get(t1).unwrap();
        assert!(task1.blocks.contains(&t2));
    }

    #[test]
    fn blocked_task_not_ready() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("First", None::<String>);
        let t2 = planner.add_task("Second", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        assert!(planner.is_ready(t1));
        assert!(!planner.is_ready(t2));
    }

    #[test]
    fn task_becomes_ready_when_deps_complete() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("First", None::<String>);
        let t2 = planner.add_task("Second", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        assert!(!planner.is_ready(t2));
        planner.start(t1).unwrap();
        planner.complete(t1).unwrap();
        assert!(planner.is_ready(t2));
    }

    #[test]
    fn multiple_dependencies() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);
        planner.add_dependency(t3, t1).unwrap();
        planner.add_dependency(t3, t2).unwrap();

        assert!(!planner.is_ready(t3));
        planner.start(t1).unwrap();
        planner.complete(t1).unwrap();
        assert!(!planner.is_ready(t3)); // still blocked by t2

        planner.start(t2).unwrap();
        planner.complete(t2).unwrap();
        assert!(planner.is_ready(t3));
    }

    #[test]
    fn self_dependency_rejected() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Task", None::<String>);
        assert!(planner.add_dependency(t1, t1).is_err());
    }

    #[test]
    fn cycle_detection() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);

        planner.add_dependency(t2, t1).unwrap(); // B depends on A
        planner.add_dependency(t3, t2).unwrap(); // C depends on B
        assert!(planner.add_dependency(t1, t3).is_err()); // A depends on C → cycle!
    }

    #[test]
    fn nonexistent_dependency_rejected() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Task", None::<String>);
        assert!(planner.add_dependency(t1, 999).is_err());
        assert!(planner.add_dependency(999, t1).is_err());
    }

    // ─── Remove ─────────────────────────────────────────────────────────

    #[test]
    fn remove_task_cleans_edges() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);
        planner.add_dependency(t2, t1).unwrap();
        planner.add_dependency(t3, t2).unwrap();

        planner.remove(t2);
        // t1 should no longer have t2 in blocks
        assert!(planner.get(t1).unwrap().blocks.is_empty());
        // t3 should no longer be blocked by t2
        assert!(planner.get(t3).unwrap().blocked_by.is_empty());
        assert!(planner.is_ready(t3));
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut planner = Planner::new();
        assert!(planner.remove(999).is_none());
    }

    // ─── Queries ────────────────────────────────────────────────────────

    #[test]
    fn ready_tasks() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);
        planner.add_dependency(t3, t1).unwrap();

        let ready = planner.ready_tasks();
        let ids: Vec<TaskId> = ready.iter().map(|t| t.id).collect();
        assert!(ids.contains(&t1));
        assert!(ids.contains(&t2));
        assert!(!ids.contains(&t3));
    }

    #[test]
    fn blocked_tasks() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        let blocked = planner.blocked_tasks();
        assert_eq!(blocked.len(), 1);
        assert_eq!(blocked[0].id, t2);
    }

    #[test]
    fn in_progress_tasks() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        planner.start(t1).unwrap();

        let in_progress = planner.in_progress_tasks();
        assert_eq!(in_progress.len(), 1);
        assert_eq!(in_progress[0].id, t1);
    }

    #[test]
    fn next_task_returns_first_ready() {
        let mut planner = Planner::new();
        let _t1 = planner.add_task("First", None::<String>);
        let _t2 = planner.add_task("Second", None::<String>);

        let next = planner.next_task().unwrap();
        assert_eq!(next.id, 1);
    }

    #[test]
    fn next_task_skips_blocked() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        planner.add_dependency(t2, t1).unwrap();
        planner.start(t1).unwrap();

        // t1 is in progress, t2 is blocked, no ready tasks
        assert!(planner.next_task().is_none());
    }

    // ─── Counts ─────────────────────────────────────────────────────────

    #[test]
    fn counts() {
        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);
        planner.start(t3).unwrap();
        planner.complete(t3).unwrap();

        let counts = planner.counts();
        assert_eq!(counts.total, 3);
        assert_eq!(counts.pending, 2);
        assert_eq!(counts.completed, 1);
    }

    #[test]
    fn is_done() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        assert!(!planner.is_done());

        planner.start(t1).unwrap();
        planner.complete(t1).unwrap();
        assert!(!planner.is_done());

        planner.skip(t2).unwrap();
        assert!(planner.is_done());
    }

    // ─── Topological Sort ───────────────────────────────────────────────

    #[test]
    fn topological_order_linear() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);
        planner.add_dependency(t2, t1).unwrap();
        planner.add_dependency(t3, t2).unwrap();

        let order = planner.topological_order().unwrap();
        let pos_a = order.iter().position(|&id| id == t1).unwrap();
        let pos_b = order.iter().position(|&id| id == t2).unwrap();
        let pos_c = order.iter().position(|&id| id == t3).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn topological_order_diamond() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Start", None::<String>);
        let t2 = planner.add_task("Left", None::<String>);
        let t3 = planner.add_task("Right", None::<String>);
        let t4 = planner.add_task("End", None::<String>);
        planner.add_dependency(t2, t1).unwrap();
        planner.add_dependency(t3, t1).unwrap();
        planner.add_dependency(t4, t2).unwrap();
        planner.add_dependency(t4, t3).unwrap();

        let order = planner.topological_order().unwrap();
        assert_eq!(order.len(), 4);
        let pos_start = order.iter().position(|&id| id == t1).unwrap();
        let pos_end = order.iter().position(|&id| id == t4).unwrap();
        assert!(pos_start < pos_end);
    }

    #[test]
    fn topological_order_independent() {
        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        planner.add_task("B", None::<String>);
        planner.add_task("C", None::<String>);

        let order = planner.topological_order().unwrap();
        assert_eq!(order.len(), 3);
    }

    // ─── Serialization ──────────────────────────────────────────────────

    #[test]
    fn snapshot_roundtrip() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", Some("Working on A"));
        let t2 = planner.add_task("B", None::<String>);
        planner.add_dependency(t2, t1).unwrap();
        planner.start(t1).unwrap();
        planner.complete(t1).unwrap();

        let json = planner.to_json().unwrap();
        let restored = Planner::from_json(&json).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored.get(t1).unwrap().status, TaskStatus::Completed);
        assert!(restored.is_ready(t2));
    }

    #[test]
    fn snapshot_equality() {
        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        planner.add_task("B", None::<String>);

        let s1 = planner.snapshot();
        let s2 = planner.snapshot();
        assert_eq!(s1, s2);
    }

    #[test]
    fn snapshot_json_pretty() {
        let mut planner = Planner::new();
        planner.add_task("Task 1", None::<String>);
        let json = planner.snapshot().to_json_pretty().unwrap();
        assert!(json.contains("Task 1"));
        assert!(json.contains('\n'));
    }

    // ─── Display ────────────────────────────────────────────────────────

    #[test]
    fn render_basic() {
        let mut planner = Planner::new();
        planner.add_task("Implement feature", Some("Implementing feature"));
        planner.add_task("Write tests", None::<String>);

        let output = planner.render();
        assert!(output.contains("◻ Implement feature"));
        assert!(output.contains("◻ Write tests"));
        assert!(output.contains("0/2 completed"));
    }

    #[test]
    fn render_with_active_task() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Build it", Some("Building it"));
        planner.add_task("Test it", None::<String>);
        planner.start(t1).unwrap();

        let output = planner.render();
        assert!(output.contains("✽ Building it…"));
        assert!(output.contains("◼ Build it"));
        assert!(output.contains("◻ Test it"));
    }

    #[test]
    fn render_with_blocked_tasks() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("First", None::<String>);
        let t2 = planner.add_task("Second", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        let output = planner.render();
        assert!(output.contains("blocked by #1"));
    }

    #[test]
    fn render_completed_tasks() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Done task", None::<String>);
        planner.start(t1).unwrap();
        planner.complete(t1).unwrap();

        let output = planner.render();
        assert!(output.contains("✓ Done task"));
        assert!(output.contains("1/1 completed"));
    }

    // ─── Status Symbols ─────────────────────────────────────────────────

    #[test]
    fn status_symbols() {
        assert_eq!(TaskStatus::Pending.symbol(), "◻");
        assert_eq!(TaskStatus::InProgress.symbol(), "◼");
        assert_eq!(TaskStatus::Completed.symbol(), "✓");
        assert_eq!(TaskStatus::Failed.symbol(), "✗");
        assert_eq!(TaskStatus::Skipped.symbol(), "⊘");
    }

    #[test]
    fn status_terminal() {
        assert!(!TaskStatus::Pending.is_terminal());
        assert!(!TaskStatus::InProgress.is_terminal());
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(TaskStatus::Skipped.is_terminal());
    }

    // ─── Duration Formatting ────────────────────────────────────────────

    #[test]
    fn format_duration_seconds() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0s");
        assert_eq!(format_duration(Duration::from_secs(45)), "45s");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(444)), "7m 24s");
    }

    #[test]
    fn format_duration_hours() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m");
    }

    // ─── Metadata ───────────────────────────────────────────────────────

    #[test]
    fn task_metadata() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        let task = planner.get_mut(id).unwrap();
        task.metadata.insert("priority".into(), "high".into());

        assert_eq!(
            planner.get(id).unwrap().metadata.get("priority"),
            Some(&"high".to_string())
        );
    }

    // ─── Blocked-by Annotation ──────────────────────────────────────────

    #[test]
    fn blocked_by_annotation_none() {
        let task = PlanTask::new(1, "Test", None);
        assert!(task.blocked_by_annotation().is_none());
    }

    #[test]
    fn blocked_by_annotation_single() {
        let mut task = PlanTask::new(2, "Test", None);
        task.blocked_by.insert(1);
        assert_eq!(task.blocked_by_annotation().unwrap(), "blocked by #1");
    }

    #[test]
    fn blocked_by_annotation_multiple() {
        let mut task = PlanTask::new(3, "Test", None);
        task.blocked_by.insert(1);
        task.blocked_by.insert(2);
        assert_eq!(task.blocked_by_annotation().unwrap(), "blocked by #1, #2");
    }

    // ─── All Tasks Order ────────────────────────────────────────────────

    #[test]
    fn all_tasks_creation_order() {
        let mut planner = Planner::new();
        let _c = planner.add_task("C", None::<String>);
        let _a = planner.add_task("A", None::<String>);
        let _b = planner.add_task("B", None::<String>);

        let all = planner.all_tasks();
        let subjects: Vec<&str> = all.iter().map(|t| t.subject.as_str()).collect();
        assert_eq!(subjects, vec!["C", "A", "B"]);
    }

    // ─── Edge Cases ─────────────────────────────────────────────────────

    #[test]
    fn empty_planner() {
        let planner = Planner::new();
        assert!(planner.is_empty());
        assert!(planner.is_done());
        assert!(planner.next_task().is_none());
        assert!(planner.ready_tasks().is_empty());
        assert!(planner.topological_order().unwrap().is_empty());

        let output = planner.render();
        assert!(output.is_empty() || output.trim().is_empty());
    }

    #[test]
    fn failed_dep_unblocks_downstream() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        planner.start(t1).unwrap();
        planner.fail(t1).unwrap();
        // Failed is terminal, so t2 is now unblocked
        assert!(planner.is_ready(t2));
    }

    #[test]
    fn skipped_dep_unblocks_downstream() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        planner.add_dependency(t2, t1).unwrap();

        planner.skip(t1).unwrap();
        assert!(planner.is_ready(t2));
    }

    // ─── Clarifying Questions ──────────────────────────────────────────

    #[test]
    fn add_question() {
        let mut planner = Planner::new();
        let qid = planner.add_question("Which database?");
        assert_eq!(qid, 1);

        let q = planner.get_question(qid).unwrap();
        assert_eq!(q.question, "Which database?");
        assert!(!q.is_answered());
        assert!(q.options.is_empty());
    }

    #[test]
    fn add_question_with_options() {
        let mut planner = Planner::new();
        let qid = planner.add_question_with_options(
            "Auth method?",
            vec!["JWT".into(), "OAuth".into(), "API key".into()],
        );

        let q = planner.get_question(qid).unwrap();
        assert_eq!(q.options.len(), 3);
        assert_eq!(q.options[1], "OAuth");
    }

    #[test]
    fn answer_question() {
        let mut planner = Planner::new();
        let qid = planner.add_question("Which database?");
        assert!(planner.has_open_questions());
        assert!(!planner.is_ready_to_execute());

        planner.answer_question(qid, "PostgreSQL").unwrap();

        let q = planner.get_question(qid).unwrap();
        assert!(q.is_answered());
        assert_eq!(q.answer.as_deref(), Some("PostgreSQL"));
        assert!(!planner.has_open_questions());
        assert!(planner.is_ready_to_execute());
    }

    #[test]
    fn answer_nonexistent_question_fails() {
        let mut planner = Planner::new();
        assert!(planner.answer_question(999, "nope").is_err());
    }

    #[test]
    fn open_questions_filters_answered() {
        let mut planner = Planner::new();
        let q1 = planner.add_question("First?");
        let _q2 = planner.add_question("Second?");
        planner.answer_question(q1, "yes").unwrap();

        let open = planner.open_questions();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].question, "Second?");
    }

    #[test]
    fn no_questions_means_ready() {
        let planner = Planner::new();
        assert!(!planner.has_open_questions());
        assert!(planner.is_ready_to_execute());
    }

    #[test]
    fn link_question_to_task() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Build auth", None::<String>);
        let qid = planner.add_question("JWT or OAuth?");
        planner.link_question_to_task(qid, t1).unwrap();

        let q = planner.get_question(qid).unwrap();
        assert!(q.related_tasks.contains(&t1));
    }

    #[test]
    fn link_nonexistent_question_fails() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Task", None::<String>);
        assert!(planner.link_question_to_task(999, t1).is_err());
    }

    #[test]
    fn question_ids_increment() {
        let mut planner = Planner::new();
        let q1 = planner.add_question("A?");
        let q2 = planner.add_question("B?");
        assert_eq!(q1, 1);
        assert_eq!(q2, 2);
    }

    #[test]
    fn all_questions() {
        let mut planner = Planner::new();
        planner.add_question("A?");
        planner.add_question("B?");
        assert_eq!(planner.all_questions().len(), 2);
    }

    #[test]
    fn questions_survive_snapshot() {
        let mut planner = Planner::new();
        let qid = planner
            .add_question_with_options("Which DB?", vec!["Postgres".into(), "SQLite".into()]);
        planner.answer_question(qid, "Postgres").unwrap();

        let json = planner.to_json().unwrap();
        let restored = Planner::from_json(&json).unwrap();

        let q = restored.get_question(qid).unwrap();
        assert_eq!(q.question, "Which DB?");
        assert_eq!(q.answer.as_deref(), Some("Postgres"));
        assert_eq!(q.options.len(), 2);
    }

    #[test]
    fn render_shows_open_questions() {
        let mut planner = Planner::new();
        planner.add_task("Build it", None::<String>);
        planner.add_question("Which framework?");

        let output = planner.render();
        assert!(output.contains("1 open question(s)"));
        assert!(output.contains("? Which framework?"));
    }

    #[test]
    fn render_shows_options() {
        let mut planner = Planner::new();
        planner.add_task("Auth", None::<String>);
        planner.add_question_with_options("Auth method?", vec!["JWT".into(), "OAuth".into()]);

        let output = planner.render();
        assert!(output.contains("[JWT / OAuth]"));
    }

    #[test]
    fn render_hides_answered_questions() {
        let mut planner = Planner::new();
        planner.add_task("Build it", None::<String>);
        let qid = planner.add_question("Which framework?");
        planner.answer_question(qid, "Axum").unwrap();

        let output = planner.render();
        assert!(!output.contains("open question"));
    }

    // ─── Retry ─────────────────────────────────────────────────────────

    #[test]
    fn retry_failed_task() {
        let mut planner = Planner::new();
        let id = planner.add_task("Flaky deploy", None::<String>);
        planner.set_max_retries(id, 2).unwrap();

        planner.start(id).unwrap();
        planner.fail(id).unwrap();
        assert_eq!(planner.get(id).unwrap().attempt, 1);

        planner.retry(id).unwrap();
        let task = planner.get(id).unwrap();
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.attempt, 2);
    }

    #[test]
    fn retry_exhausted() {
        let mut planner = Planner::new();
        let id = planner.add_task("Flaky", None::<String>);
        planner.set_max_retries(id, 1).unwrap();

        // Attempt 1: fail
        planner.start(id).unwrap();
        planner.fail(id).unwrap();
        // Retry → attempt 2
        planner.retry(id).unwrap();
        planner.start(id).unwrap();
        planner.fail(id).unwrap();
        // No more retries (attempt 2 > max_retries 1)
        assert!(planner.retry(id).is_err());
    }

    #[test]
    fn retry_non_failed_task_fails() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        planner.set_max_retries(id, 3).unwrap();
        assert!(planner.retry(id).is_err()); // pending, not failed
    }

    #[test]
    fn retry_without_max_retries_fails() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        // max_retries defaults to 0
        planner.start(id).unwrap();
        planner.fail(id).unwrap();
        assert!(planner.retry(id).is_err());
    }

    #[test]
    fn can_retry_check() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        planner.set_max_retries(id, 1).unwrap();

        assert!(!planner.get(id).unwrap().can_retry()); // pending
        planner.start(id).unwrap();
        planner.fail(id).unwrap();
        assert!(planner.get(id).unwrap().can_retry()); // failed, attempt 1 <= max 1
    }

    #[test]
    fn retry_all_retries_eligible() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);

        planner.set_max_retries(t1, 2).unwrap();
        planner.set_max_retries(t2, 0).unwrap(); // no retry
        planner.set_max_retries(t3, 1).unwrap();

        planner.start(t1).unwrap();
        planner.fail(t1).unwrap();
        planner.start(t2).unwrap();
        planner.fail(t2).unwrap();
        planner.start(t3).unwrap();
        planner.fail(t3).unwrap();

        let retried = planner.retry_all();
        assert_eq!(retried.len(), 2); // t1 and t3
        assert!(retried.contains(&t1));
        assert!(retried.contains(&t3));
        assert_eq!(planner.get(t2).unwrap().status, TaskStatus::Failed); // stayed failed
    }

    #[test]
    fn fail_with_error_records_message() {
        let mut planner = Planner::new();
        let id = planner.add_task("Deploy", None::<String>);
        planner.start(id).unwrap();
        planner
            .fail_with_error(id, "Connection refused on port 443")
            .unwrap();

        let task = planner.get(id).unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(
            task.last_error.as_deref(),
            Some("Connection refused on port 443")
        );
    }

    #[test]
    fn retry_preserves_last_error() {
        let mut planner = Planner::new();
        let id = planner.add_task("Deploy", None::<String>);
        planner.set_max_retries(id, 1).unwrap();
        planner.start(id).unwrap();
        planner.fail_with_error(id, "timeout").unwrap();
        planner.retry(id).unwrap();

        // Error from previous attempt is preserved for context
        assert_eq!(
            planner.get(id).unwrap().last_error.as_deref(),
            Some("timeout")
        );
    }

    // ─── Resume ────────────────────────────────────────────────────────

    #[test]
    fn resume_resets_in_progress_to_pending() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("A", None::<String>);
        let t2 = planner.add_task("B", None::<String>);
        let t3 = planner.add_task("C", None::<String>);

        planner.start(t1).unwrap();
        planner.complete(t1).unwrap();
        planner.start(t2).unwrap(); // in progress when "crash" happens

        let resumed = planner.resume();
        assert_eq!(resumed, vec![t2]);
        assert_eq!(planner.get(t1).unwrap().status, TaskStatus::Completed); // untouched
        assert_eq!(planner.get(t2).unwrap().status, TaskStatus::Pending); // reset
        assert_eq!(planner.get(t2).unwrap().attempt, 2); // incremented
        assert_eq!(planner.get(t3).unwrap().status, TaskStatus::Pending); // untouched
    }

    #[test]
    fn resume_no_in_progress_returns_empty() {
        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        let resumed = planner.resume();
        assert!(resumed.is_empty());
    }

    #[test]
    fn resume_after_snapshot_roundtrip() {
        let mut planner = Planner::new();
        let t1 = planner.add_task("Build", None::<String>);
        let _t2 = planner.add_task("Test", None::<String>);
        planner.start(t1).unwrap();

        // Simulate crash: serialize, restore, resume
        let json = planner.to_json().unwrap();
        let mut restored = Planner::from_json(&json).unwrap();
        let resumed = restored.resume();

        assert_eq!(resumed, vec![t1]);
        assert_eq!(restored.get(t1).unwrap().status, TaskStatus::Pending);
        assert!(restored.is_ready(t1));
    }

    // ─── Checkpoint ────────────────────────────────────────────────────

    #[test]
    fn checkpoint_stores_output() {
        let mut planner = Planner::new();
        let id = planner.add_task("Generate code", None::<String>);
        planner.start(id).unwrap();
        planner
            .checkpoint(id, "fn main() { /* partial */ }")
            .unwrap();

        assert_eq!(
            planner.get(id).unwrap().output.as_deref(),
            Some("fn main() { /* partial */ }")
        );
    }

    #[test]
    fn checkpoint_survives_snapshot() {
        let mut planner = Planner::new();
        let id = planner.add_task("Generate", None::<String>);
        planner.start(id).unwrap();
        planner.checkpoint(id, "partial output").unwrap();

        let json = planner.to_json().unwrap();
        let restored = Planner::from_json(&json).unwrap();
        assert_eq!(
            restored.get(id).unwrap().output.as_deref(),
            Some("partial output")
        );
    }

    #[test]
    fn checkpoint_overwrites_previous() {
        let mut planner = Planner::new();
        let id = planner.add_task("Build", None::<String>);
        planner.start(id).unwrap();
        planner.checkpoint(id, "v1").unwrap();
        planner.checkpoint(id, "v2").unwrap();
        assert_eq!(planner.get(id).unwrap().output.as_deref(), Some("v2"));
    }

    #[test]
    fn checkpoint_nonexistent_task_fails() {
        let mut planner = Planner::new();
        assert!(planner.checkpoint(999, "data").is_err());
    }

    #[test]
    fn checkpoint_preserved_across_retry() {
        let mut planner = Planner::new();
        let id = planner.add_task("Deploy", None::<String>);
        planner.set_max_retries(id, 1).unwrap();
        planner.start(id).unwrap();
        planner.checkpoint(id, "uploaded 3/5 files").unwrap();
        planner.fail_with_error(id, "network error").unwrap();
        planner.retry(id).unwrap();

        // Both output and error preserved for the retry attempt
        let task = planner.get(id).unwrap();
        assert_eq!(task.output.as_deref(), Some("uploaded 3/5 files"));
        assert_eq!(task.last_error.as_deref(), Some("network error"));
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.attempt, 2);
    }

    #[test]
    fn render_shows_retry_count() {
        let mut planner = Planner::new();
        let id = planner.add_task("Flaky deploy", None::<String>);
        planner.set_max_retries(id, 2).unwrap();
        planner.start(id).unwrap();
        planner.fail(id).unwrap();

        let output = planner.render();
        assert!(output.contains("(attempt 1/3)"));
    }

    // ─── Set Max Retries ───────────────────────────────────────────────

    #[test]
    fn set_max_retries() {
        let mut planner = Planner::new();
        let id = planner.add_task("Task", None::<String>);
        assert_eq!(planner.get(id).unwrap().max_retries, 0);

        planner.set_max_retries(id, 5).unwrap();
        assert_eq!(planner.get(id).unwrap().max_retries, 5);
    }

    #[test]
    fn set_max_retries_nonexistent_fails() {
        let mut planner = Planner::new();
        assert!(planner.set_max_retries(999, 3).is_err());
    }
}
