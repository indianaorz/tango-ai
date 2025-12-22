// tango-pvp/src/telemetry.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameTelemetry {
    V1(FrameTelemetryV1),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTelemetryV1 {
    // --- Dynamic Battle State (RAM) ---
    pub player_health: Option<u16>,
    pub enemy_health: Option<u16>,
    pub player_pos: Option<(u16, u16)>,
    pub enemy_pos: Option<(u16, u16)>,
    pub player_charge: Option<u16>,
    pub enemy_charge: Option<u16>,
    pub player_chip: Option<u16>,
    pub enemy_chip: Option<u16>,
    pub cust_gauge: Option<u16>,
    pub inside_window: Option<bool>,
    
    pub grid_state: Option<Vec<u16>>,
    pub grid_owner_state: Option<Vec<u16>>,
    
    // The chips currently available on the Custom Screen (RAM)
    pub player_hand: Option<Vec<u16>>, 

    // --- Static Save Data (Globals) ---
    // The full deck (30 chips)
    pub player_folder: Option<Vec<u16>>,
    pub enemy_folder: Option<Vec<u16>>,
    
    // The codes for the full deck
    pub player_code_folder: Option<Vec<u16>>,
    pub enemy_code_folder: Option<Vec<u16>>,
    
    pub player_tag_chips: Option<Vec<u16>>,
    pub enemy_tag_chips: Option<Vec<u16>>,
    pub player_reg_chip: Option<u16>,
    pub enemy_reg_chip: Option<u16>,
    pub player_navi_cust: Option<Vec<usize>>,
    pub enemy_navi_cust: Option<Vec<usize>>,
}