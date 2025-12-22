use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameTelemetry {
    V1(FrameTelemetryV1),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameTelemetryV1 {
    // --- Dynamic RAM ---
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
    
    // --- Emotions ---
    pub player_emotion: Option<u16>, // Window Portrait
    pub enemy_emotion: Option<u16>,
    
    // [FIX] Added Game Emotions (Grid Status like Full Synchro)
    pub player_game_emotion: Option<u16>, 
    pub enemy_game_emotion: Option<u16>,

    pub beast_mode: Option<u16>,
    pub cross_window: Option<u16>,
    
    // --- Chip Selection ---
    pub chip_select_count: Option<u16>,
    pub chip_visible_count: Option<u16>,
    pub selected_menu_index: Option<u16>,
    pub selected_cross_index: Option<u16>,

    // --- Complex Data ---
    pub grid_state: Option<Vec<u16>>,
    pub grid_owner_state: Option<Vec<u16>>,
    
    // [FIX] Split Hand into Slots (IDs) and Codes
    pub chip_slots: Option<Vec<u16>>,
    pub chip_codes: Option<Vec<u16>>,
    // Deprecating merged 'player_hand' to match ScreenImageDetails
    // pub player_hand: Option<Vec<u16>>, 
    
    pub selected_chip_indices: Option<Vec<u16>>,

    // --- Static Save Data ---
    pub player_folder: Option<Vec<u16>>,
    pub enemy_folder: Option<Vec<u16>>,
    pub player_code_folder: Option<Vec<u16>>,
    pub enemy_code_folder: Option<Vec<u16>>,
    pub player_tag_chips: Option<Vec<u16>>,
    pub enemy_tag_chips: Option<Vec<u16>>,
    pub player_reg_chip: Option<u16>,
    pub enemy_reg_chip: Option<u16>,
    pub player_navi_cust: Option<Vec<usize>>,
    pub enemy_navi_cust: Option<Vec<usize>>,
}