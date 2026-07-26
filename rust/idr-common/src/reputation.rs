use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::Ipv4Addr;

/// IP/ASN reputation database for classifying trust levels
#[derive(Debug, Clone)]
pub struct ReputationDb {
    /// ASN → trust classification
    asn_trust: HashMap<String, TrustLevel>,
    /// CIDR prefix → ASN mapping (simplified)
    prefix_to_asn: Vec<(Ipv4Addr, u8, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    /// Major cloud/CDN providers (Google, Meta, Cloudflare, AWS)
    HighTrust,
    /// Known commercial ISPs
    CommercialIsp,
    /// Residential ISPs — suspicious in PTR reversal context
    ResidentialIsp,
    /// Unknown/unclassified
    Unknown,
    /// Known malicious infrastructure
    Malicious,
}

impl ReputationDb {
    pub fn new() -> Self {
        let mut db = Self {
            asn_trust: HashMap::new(),
            prefix_to_asn: Vec::new(),
        };
        db.load_defaults();
        db
    }

    fn load_defaults(&mut self) {
        // High-trust: Google
        self.add_asn("AS15169", TrustLevel::HighTrust);
        self.add_prefix("8.8.8.0", 24, "AS15169");
        self.add_prefix("142.250.0.0", 15, "AS15169");
        self.add_prefix("172.217.0.0", 16, "AS15169");
        self.add_prefix("216.58.0.0", 16, "AS15169");

        // High-trust: Meta
        self.add_asn("AS32934", TrustLevel::HighTrust);
        self.add_prefix("157.240.0.0", 16, "AS32934");
        self.add_prefix("31.13.0.0", 16, "AS32934");

        // High-trust: Cloudflare
        self.add_asn("AS13335", TrustLevel::HighTrust);
        self.add_prefix("1.1.1.0", 24, "AS13335");
        self.add_prefix("104.16.0.0", 12, "AS13335");

        // Residential ISPs (suspicious in reversal context)
        self.add_asn("AS3320", TrustLevel::ResidentialIsp); // Deutsche Telekom
        self.add_asn("AS45758", TrustLevel::ResidentialIsp); // 3BB Thailand
        self.add_asn("AS7922", TrustLevel::ResidentialIsp); // Comcast
        self.add_asn("AS22773", TrustLevel::ResidentialIsp); // Cox
        self.add_asn("AS209", TrustLevel::ResidentialIsp); // CenturyLink
    }

    pub fn add_asn(&mut self, asn: &str, trust: TrustLevel) {
        self.asn_trust.insert(asn.to_string(), trust);
    }

    pub fn add_prefix(&mut self, ip: &str, prefix_len: u8, asn: &str) {
        if prefix_len > 32 {
            return; // Invalid CIDR prefix length for IPv4
        }
        if let Ok(addr) = ip.parse::<Ipv4Addr>() {
            self.prefix_to_asn
                .push((addr, prefix_len, asn.to_string()));
        }
    }

    /// Look up the ASN for a given IPv4 address using longest-prefix match
    pub fn lookup_asn(&self, ip: &Ipv4Addr) -> Option<&str> {
        let ip_bits = u32::from(*ip);
        let mut best_match: Option<(&str, u8)> = None;

        for (prefix_ip, prefix_len, asn) in &self.prefix_to_asn {
            if *prefix_len > 32 {
                continue; // Skip invalid entries
            }
            let prefix_bits = u32::from(*prefix_ip);
            let mask = if *prefix_len == 0 {
                0
            } else {
                !0u32 << (32 - prefix_len)
            };

            if (ip_bits & mask) == (prefix_bits & mask) {
                match best_match {
                    None => best_match = Some((asn.as_str(), *prefix_len)),
                    Some((_, best_len)) if *prefix_len > best_len => {
                        best_match = Some((asn.as_str(), *prefix_len));
                    }
                    _ => {}
                }
            }
        }

        best_match.map(|(asn, _)| asn)
    }

    /// Get the trust level for an ASN
    pub fn trust_level(&self, asn: &str) -> TrustLevel {
        self.asn_trust
            .get(asn)
            .copied()
            .unwrap_or(TrustLevel::Unknown)
    }

    /// Classify an IP address's trust level
    pub fn classify_ip(&self, ip: &Ipv4Addr) -> TrustLevel {
        match self.lookup_asn(ip) {
            Some(asn) => self.trust_level(asn),
            None => TrustLevel::Unknown,
        }
    }

    /// Check if a forward IP is high-trust but its reversed form is residential
    /// This is the core of the Octet Reversal (DPI Evasion) detection
    pub fn check_octet_reversal(&self, forward_ip: &Ipv4Addr) -> Option<OctetReversalResult> {
        let octets = forward_ip.octets();
        let reversed = Ipv4Addr::new(octets[3], octets[2], octets[1], octets[0]);

        let forward_trust = self.classify_ip(forward_ip);
        let reversed_trust = self.classify_ip(&reversed);

        if forward_trust == TrustLevel::HighTrust && reversed_trust == TrustLevel::ResidentialIsp {
            let forward_asn = self.lookup_asn(forward_ip).unwrap_or("unknown").to_string();
            let reversed_asn = self.lookup_asn(&reversed).unwrap_or("unknown").to_string();

            Some(OctetReversalResult {
                forward_ip: *forward_ip,
                reversed_ip: reversed,
                forward_asn,
                reversed_asn,
                forward_trust,
                reversed_trust,
            })
        } else {
            None
        }
    }
}

impl Default for ReputationDb {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OctetReversalResult {
    pub forward_ip: Ipv4Addr,
    pub reversed_ip: Ipv4Addr,
    pub forward_asn: String,
    pub reversed_asn: String,
    pub forward_trust: TrustLevel,
    pub reversed_trust: TrustLevel,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asn_lookup_google() {
        let db = ReputationDb::new();
        // 8.8.8.8 is in the 8.8.8.0/24 prefix → AS15169
        let ip: Ipv4Addr = "8.8.8.8".parse().unwrap();
        let asn = db.lookup_asn(&ip);
        assert_eq!(asn, Some("AS15169"));

        // 142.250.80.46 is in the 142.250.0.0/15 prefix → AS15169
        let ip2: Ipv4Addr = "142.250.80.46".parse().unwrap();
        assert_eq!(db.lookup_asn(&ip2), Some("AS15169"));
    }

    #[test]
    fn test_asn_lookup_meta() {
        let db = ReputationDb::new();
        // 157.240.1.35 is in the 157.240.0.0/16 prefix → AS32934
        let ip: Ipv4Addr = "157.240.1.35".parse().unwrap();
        assert_eq!(db.lookup_asn(&ip), Some("AS32934"));

        // 31.13.80.1 is in the 31.13.0.0/16 prefix → AS32934
        let ip2: Ipv4Addr = "31.13.80.1".parse().unwrap();
        assert_eq!(db.lookup_asn(&ip2), Some("AS32934"));
    }

    #[test]
    fn test_classify_ip_high_trust() {
        let db = ReputationDb::new();
        // Google DNS
        let google: Ipv4Addr = "8.8.8.8".parse().unwrap();
        assert_eq!(db.classify_ip(&google), TrustLevel::HighTrust);

        // Meta
        let meta: Ipv4Addr = "157.240.1.35".parse().unwrap();
        assert_eq!(db.classify_ip(&meta), TrustLevel::HighTrust);

        // Cloudflare
        let cf: Ipv4Addr = "1.1.1.1".parse().unwrap();
        assert_eq!(db.classify_ip(&cf), TrustLevel::HighTrust);
    }

    #[test]
    fn test_classify_ip_unknown() {
        let db = ReputationDb::new();
        // Random IPs that don't match any known prefix
        let random1: Ipv4Addr = "93.184.216.34".parse().unwrap();
        assert_eq!(db.classify_ip(&random1), TrustLevel::Unknown);

        let random2: Ipv4Addr = "203.0.113.50".parse().unwrap();
        assert_eq!(db.classify_ip(&random2), TrustLevel::Unknown);
    }

    #[test]
    fn test_octet_reversal_detection() {
        // We need an IP where:
        //   forward = HighTrust (Google)
        //   reversed octets = ResidentialIsp
        //
        // For this to work, we add a residential ISP prefix that matches
        // the reversed form of a Google IP.
        //
        // Take a Google IP in 142.250.x.y — reversed is y.x.250.142
        // We'll craft a scenario: forward_ip = 142.250.3.22
        //   reversed = 22.3.250.142
        // Add a residential prefix covering 22.0.0.0/8 mapped to a residential ASN.
        let mut db = ReputationDb::new();
        db.add_asn("AS99999", TrustLevel::ResidentialIsp);
        db.add_prefix("22.0.0.0", 8, "AS99999");

        let forward_ip: Ipv4Addr = "142.250.3.22".parse().unwrap();
        // Forward should be HighTrust (Google 142.250.0.0/15)
        assert_eq!(db.classify_ip(&forward_ip), TrustLevel::HighTrust);

        // Reversed = 22.3.250.142 should be ResidentialIsp (22.0.0.0/8)
        let reversed: Ipv4Addr = "22.3.250.142".parse().unwrap();
        assert_eq!(db.classify_ip(&reversed), TrustLevel::ResidentialIsp);

        // check_octet_reversal should detect this
        let result = db.check_octet_reversal(&forward_ip);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.forward_ip, forward_ip);
        assert_eq!(r.reversed_ip, reversed);
        assert_eq!(r.forward_trust, TrustLevel::HighTrust);
        assert_eq!(r.reversed_trust, TrustLevel::ResidentialIsp);
    }
}
