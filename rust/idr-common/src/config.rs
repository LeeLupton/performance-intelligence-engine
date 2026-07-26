use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Global configuration for the IDR Sentinel Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct IdrConfig {
    pub kernel: KernelConfig,
    pub network: NetworkConfig,
    pub hardware: HardwareConfig,
    pub sentinel: SentinelConfig,
    pub dashboard: DashboardConfig,
}

impl IdrConfig {
    /// Validate all configuration values. Call after deserialization from untrusted input.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Kernel
        if self.kernel.igmp_correlation_window_ms == 0 {
            return Err(ConfigError("igmp_correlation_window_ms must be > 0".into()));
        }
        if self.kernel.suspicious_rtt_ms <= 0.0 {
            return Err(ConfigError("suspicious_rtt_ms must be > 0".into()));
        }

        // Network
        if self.network.ntp_shift_threshold_secs <= 0.0 {
            return Err(ConfigError("ntp_shift_threshold_secs must be > 0".into()));
        }
        if self.network.tls_flag_count_after_ntp == 0 {
            return Err(ConfigError("tls_flag_count_after_ntp must be > 0".into()));
        }
        if self.network.zeek_socket_path.contains("..") {
            return Err(ConfigError("zeek_socket_path must not contain '..'".into()));
        }

        // Hardware
        if self.hardware.nvme_baseline_latency_us == 0 {
            return Err(ConfigError("nvme_baseline_latency_us must be > 0".into()));
        }
        if self.hardware.nvme_deviation_threshold_pct <= 0.0 {
            return Err(ConfigError("nvme_deviation_threshold_pct must be > 0".into()));
        }
        if self.hardware.mac_flap_threshold == 0 {
            return Err(ConfigError("mac_flap_threshold must be > 0".into()));
        }
        if self.hardware.mac_flap_window_secs == 0 {
            return Err(ConfigError("mac_flap_window_secs must be > 0".into()));
        }

        // Sentinel — validate interface name (alphanumeric, dashes, dots, colons)
        if !is_valid_interface_name(&self.sentinel.panic_interface) {
            return Err(ConfigError(format!(
                "panic_interface '{}' contains invalid characters (allowed: alphanumeric, dash, dot, colon)",
                self.sentinel.panic_interface
            )));
        }
        self.sentinel.ws_listen_addr.parse::<SocketAddr>().map_err(|_| {
            ConfigError(format!("invalid ws_listen_addr: '{}'", self.sentinel.ws_listen_addr))
        })?;

        // Dashboard
        self.dashboard.listen_addr.parse::<SocketAddr>().map_err(|_| {
            ConfigError(format!("invalid dashboard listen_addr: '{}'", self.dashboard.listen_addr))
        })?;

        Ok(())
    }
}

/// Validate that a network interface name contains only safe characters
fn is_valid_interface_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 15 // IFNAMSIZ - 1
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == ':' || c == '_')
}

/// Validate that an NVMe device path looks like a real device path
pub fn is_valid_device_path(path: &str) -> bool {
    path.starts_with("/dev/")
        && !path.contains("..")
        && path.chars().all(|c| c.is_ascii_alphanumeric() || c == '/' || c == '-' || c == '_')
}

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "config validation error: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// IGMP → QUIC correlation window in milliseconds
    pub igmp_correlation_window_ms: u64,
    /// Interface to attach XDP program to
    pub xdp_interface: String,
    /// High-trust IP prefixes (Google, Meta, etc.)
    pub high_trust_asn_prefixes: Vec<String>,
    /// TTL threshold for physics anomaly detection
    pub suspicious_ttl: u8,
    /// RTT threshold (ms) for physics anomaly detection
    pub suspicious_rtt_ms: f64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            igmp_correlation_window_ms: 500,
            xdp_interface: "eth0".to_string(),
            high_trust_asn_prefixes: vec![
                // Google
                "8.8.8.0/24".into(),
                "142.250.0.0/15".into(),
                "172.217.0.0/16".into(),
                "216.58.0.0/16".into(),
                // Meta/Facebook
                "157.240.0.0/16".into(),
                "31.13.0.0/16".into(),
                "179.60.192.0/22".into(),
            ],
            suspicious_ttl: 63,
            suspicious_rtt_ms: 5.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Path to Zeek Unix socket for JSON log ingestion
    pub zeek_socket_path: String,
    /// NTP time-shift threshold in seconds
    pub ntp_shift_threshold_secs: f64,
    /// Number of TLS handshakes to flag after NTP time-shift
    pub tls_flag_count_after_ntp: usize,
    /// Residential ISP ASNs for octet reversal detection
    pub residential_asns: Vec<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            zeek_socket_path: "/var/run/idr/zeek.sock".to_string(),
            ntp_shift_threshold_secs: 300.0,
            tls_flag_count_after_ntp: 10,
            residential_asns: vec![
                "AS3320".into(),  // Deutsche Telekom
                "AS45758".into(), // Triple T Broadband (3BB)
                "AS7922".into(),  // Comcast
                "AS22773".into(), // Cox
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// NVMe device to monitor
    pub nvme_device: String,
    /// I/O latency deviation threshold (percentage)
    pub nvme_deviation_threshold_pct: f64,
    /// Baseline I/O latency in microseconds (manufacturer spec)
    pub nvme_baseline_latency_us: u64,
    /// Gateway IP for ARP/MAC monitoring
    pub gateway_ip: String,
    /// MAC flap threshold (changes per window)
    pub mac_flap_threshold: u32,
    /// MAC flap detection window in seconds
    pub mac_flap_window_secs: u64,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            nvme_device: "/dev/nvme0".to_string(),
            nvme_deviation_threshold_pct: 15.0,
            nvme_baseline_latency_us: 100,
            gateway_ip: "192.168.1.1".to_string(),
            mac_flap_threshold: 3,
            mac_flap_window_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SentinelConfig {
    /// Network interface to kill on panic
    pub panic_interface: String,
    /// Allow NVMe crypto-erase on panic (DESTRUCTIVE)
    pub allow_nvme_erase: bool,
    /// Enable automatic panic response
    pub auto_panic_enabled: bool,
    /// WebSocket listen address for dashboard
    pub ws_listen_addr: String,
    /// Optional shared-secret required as `?token=` on /ws/events.
    /// Empty (default) disables auth — same behavior as before. Set
    /// to a non-empty random string on shared hosts to require it.
    /// HTTP /api/* routes are NOT gated by this token (use a reverse-
    /// proxy basic-auth or IP allow-list for HTTP if needed).
    #[serde(default)]
    pub ws_auth_token: String,
}

impl Default for SentinelConfig {
    fn default() -> Self {
        Self {
            panic_interface: "eth0".to_string(),
            allow_nvme_erase: false,
            auto_panic_enabled: false,
            ws_listen_addr: "127.0.0.1:9700".to_string(),
            ws_auth_token: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// HTTP listen address for the dashboard API
    pub listen_addr: String,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:9701".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_valid() {
        let config = IdrConfig::default();

        // Kernel config defaults
        assert_eq!(config.kernel.igmp_correlation_window_ms, 500);
        assert_eq!(config.kernel.xdp_interface, "eth0");
        assert!(!config.kernel.high_trust_asn_prefixes.is_empty());
        assert_eq!(config.kernel.suspicious_ttl, 63);
        assert!(config.kernel.suspicious_rtt_ms > 0.0);

        // Network config defaults
        assert!(!config.network.zeek_socket_path.is_empty());
        assert_eq!(config.network.ntp_shift_threshold_secs, 300.0);
        assert_eq!(config.network.tls_flag_count_after_ntp, 10);
        assert!(!config.network.residential_asns.is_empty());

        // Hardware config defaults
        assert!(!config.hardware.nvme_device.is_empty());
        assert!(config.hardware.nvme_deviation_threshold_pct > 0.0);

        // Sentinel config defaults
        assert!(!config.sentinel.auto_panic_enabled);
        assert!(!config.sentinel.allow_nvme_erase);
        assert!(!config.sentinel.ws_listen_addr.is_empty());

        // Dashboard config defaults
        assert!(!config.dashboard.listen_addr.is_empty());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = IdrConfig::default();

        // Serialize to JSON
        let json = serde_json::to_string(&config).expect("serialization failed");

        // Deserialize back
        let deserialized: IdrConfig =
            serde_json::from_str(&json).expect("deserialization failed");

        // Verify key fields match (no PartialEq derive, so check field by field)
        assert_eq!(
            deserialized.kernel.igmp_correlation_window_ms,
            config.kernel.igmp_correlation_window_ms
        );
        assert_eq!(
            deserialized.kernel.xdp_interface,
            config.kernel.xdp_interface
        );
        assert_eq!(
            deserialized.kernel.suspicious_ttl,
            config.kernel.suspicious_ttl
        );
        assert_eq!(
            deserialized.network.ntp_shift_threshold_secs,
            config.network.ntp_shift_threshold_secs
        );
        assert_eq!(
            deserialized.network.tls_flag_count_after_ntp,
            config.network.tls_flag_count_after_ntp
        );
        assert_eq!(
            deserialized.sentinel.auto_panic_enabled,
            config.sentinel.auto_panic_enabled
        );
        assert_eq!(
            deserialized.sentinel.allow_nvme_erase,
            config.sentinel.allow_nvme_erase
        );
        assert_eq!(
            deserialized.dashboard.listen_addr,
            config.dashboard.listen_addr
        );
        assert_eq!(
            deserialized.hardware.nvme_device,
            config.hardware.nvme_device
        );

        // Also verify re-serialization produces the same JSON
        let json2 = serde_json::to_string(&deserialized).expect("re-serialization failed");
        assert_eq!(json, json2);
    }

    #[test]
    fn test_default_config_passes_validation() {
        let config = IdrConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_rejects_zero_correlation_window() {
        let mut config = IdrConfig::default();
        config.kernel.igmp_correlation_window_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_invalid_interface() {
        let mut config = IdrConfig::default();
        config.sentinel.panic_interface = "; rm -rf /".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_rejects_invalid_listen_addr() {
        let mut config = IdrConfig::default();
        config.sentinel.ws_listen_addr = "not-a-socket-addr".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_valid_device_path() {
        assert!(is_valid_device_path("/dev/nvme0n1"));
        assert!(is_valid_device_path("/dev/sda"));
        assert!(!is_valid_device_path("../etc/passwd"));
        assert!(!is_valid_device_path("/dev/../etc/passwd"));
        assert!(!is_valid_device_path("/tmp/evil"));
    }
}
