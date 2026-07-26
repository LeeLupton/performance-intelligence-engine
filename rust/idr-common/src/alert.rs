use crate::events::{IdrEvent, Severity};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An alert is a promoted event that requires operator attention or triggers
/// automated response (panic mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub severity: Severity,
    pub title: String,
    pub description: String,
    pub source_events: Vec<Uuid>,
    pub state: AlertState,
    pub response: Option<PanicAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertState {
    Active,
    Acknowledged,
    Resolved,
    PanicTriggered,
}

/// Automated response actions the Sentinel Engine can take
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanicAction {
    /// Disable network interface immediately
    NetworkKill { interface: String },
    /// NVMe cryptographic erase (requires user toggle)
    NvmeCryptoErase { device: String },
    /// Both: network kill + crypto erase
    FullPanic {
        interface: String,
        nvme_device: String,
    },
}

impl Alert {
    pub fn from_event(event: &IdrEvent, title: String, description: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            severity: event.severity,
            title,
            description,
            source_events: vec![event.id],
            state: AlertState::Active,
            response: None,
        }
    }

    pub fn critical(title: String, description: String, source_events: Vec<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            severity: Severity::Critical,
            title,
            description,
            source_events,
            state: AlertState::Active,
            response: None,
        }
    }

    pub fn impossible(title: String, description: String, source_events: Vec<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            severity: Severity::Impossible,
            title,
            description,
            source_events,
            state: AlertState::Active,
            response: None,
        }
    }

    pub fn with_panic(mut self, action: PanicAction) -> Self {
        self.state = AlertState::PanicTriggered;
        self.response = Some(action);
        self
    }
}
