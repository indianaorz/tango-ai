// src/global.rs

use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Arc;
use egui::ColorImage;
use serde::Serialize;

// Assuming RewardPunishment is defined like this:
#[derive(Debug, Clone, Serialize)]
pub struct RewardPunishment {
    pub damage: u16,
}

// Define the structure to hold winner state
#[derive(Debug, Clone)]
pub struct WinnerState {
    pub player_won: Option<bool>,
}

lazy_static! {
    // Global variable for storing rewards
    pub static ref REWARDS: Arc<Mutex<Vec<RewardPunishment>>> = Arc::new(Mutex::new(Vec::new()));

    // Global variable for storing punishments
    pub static ref PUNISHMENTS: Arc<Mutex<Vec<RewardPunishment>>> = Arc::new(Mutex::new(Vec::new()));

    pub static ref SCREEN_IMAGE: Arc<Mutex<Option<ColorImage>>> = Arc::new(Mutex::new(None));

    // Global variable for storing local input events
    pub static ref LOCAL_INPUT: Arc<Mutex<Option<u16>>> = Arc::new(Mutex::new(None));

    pub static ref REPLAY_PATH: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    pub static ref WINNER_STATE: Arc<Mutex<WinnerState>> = Arc::new(Mutex::new(WinnerState { player_won: None }));

    // New globals for additional data points
    pub static ref PLAYER_POSITION: Arc<Mutex<Option<(u16, u16)>>> = Arc::new(Mutex::new(None));
    pub static ref ENEMY_POSITION: Arc<Mutex<Option<(u16, u16)>>> = Arc::new(Mutex::new(None));
    pub static ref PLAYER_FORMS_USED: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
    pub static ref ENEMY_FORMS_USED: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
    pub static ref IS_PLAYER_INSIDE_WINDOW: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None)); // New flag

    // Player and enemy health
    pub static ref PLAYER_HEALTH: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));
    pub static ref ENEMY_HEALTH: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Player health index
    pub static ref PLAYER_HEALTH_INDEX: Arc<Mutex<u16>> = Arc::new(Mutex::new(2));

    // Frame count
    pub static ref FRAME_COUNT: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Player charge
    pub static ref PLAYER_CHARGE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Enemy charge
    pub static ref ENEMY_CHARGE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Player selected chip number
    pub static ref PLAYER_SELECTED_CHIP: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Enemy selected chip number
    pub static ref ENEMY_SELECTED_CHIP: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Player emotion state
    pub static ref PLAYER_EMOTION_STATE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Enemy emotion state
    pub static ref ENEMY_EMOTION_STATE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    //in game player emotion state
    pub static ref PLAYER_GAME_EMOTION_STATE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));
    pub static ref ENEMY_GAME_EMOTION_STATE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Selected menu index
    pub static ref SELECTED_MENU_INDEX: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Selected cross index
    pub static ref SELECTED_CROSS_INDEX: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Chip selected count
    pub static ref CHIP_SELECTED_COUNT: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref CHIP_COUNT_VISIBLE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Chip slots
    pub static ref CHIP_SLOTS: Vec<Arc<Mutex<u16>>> = (0..8).map(|_| Arc::new(Mutex::new(0))).collect();

    // Chip codes
    pub static ref CHIP_CODES: Vec<Arc<Mutex<u16>>> = (0..8).map(|_| Arc::new(Mutex::new(0))).collect();

    
    pub static ref PLAYER_CHIP_FOLDER:Vec<Arc<Mutex<u16>>> = (0..30).map(|_| Arc::new(Mutex::new(0))).collect();
    pub static ref PLAYER_CODE_FOLDER:Vec<Arc<Mutex<u16>>> = (0..30).map(|_| Arc::new(Mutex::new(0))).collect();

    pub static ref ENEMY_CHIP_FOLDER:Vec<Arc<Mutex<u16>>> = (0..30).map(|_| Arc::new(Mutex::new(0))).collect();
    pub static ref ENEMY_CODE_FOLDER:Vec<Arc<Mutex<u16>>> = (0..30).map(|_| Arc::new(Mutex::new(0))).collect();


    pub static ref PLAYER_TAG_FOLDER:Vec<Arc<Mutex<u16>>> = (0..2).map(|_| Arc::new(Mutex::new(0))).collect();
    pub static ref ENEMY_TAG_FOLDER:Vec<Arc<Mutex<u16>>> = (0..2).map(|_| Arc::new(Mutex::new(0))).collect();

    pub static ref PLAYER_REG_CHIP: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));
    pub static ref ENEMY_REG_CHIP: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // 5 possible ids for selected chips
    pub static ref SELECTED_CHIP_INDICES: Vec<Arc<Mutex<u16>>> = (0..5).map(|_| Arc::new(Mutex::new(0))).collect();

    pub static ref BEAST_OUT_SELECTABLE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref INSIDE_CROSS_WINDOW: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    //6x3 grid of uint16s for the grid state
    pub static ref GRID_STATE: Vec<Arc<Mutex<u16>>> = (0..18).map(|_| Arc::new(Mutex::new(0))).collect();
    
    //6x3 grid of uint16s for the grid owner state
    pub static ref GRID_OWNER_STATE: Vec<Arc<Mutex<u16>>> = (0..18).map(|_| Arc::new(Mutex::new(0))).collect();

    //6x3 grid of ids which occupy the grid
    pub static ref GRID_OCCUPY_STATE: Vec<Arc<Mutex<u16>>> = (0..18).map(|_| Arc::new(Mutex::new(0))).collect();

    pub static ref PLAYER_INVIS:Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref ENEMY_INVIS: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref PLAYER_TRAP:Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref ENEMY_TRAP:Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref TIME_FROZEN: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref PLAYER_AURA: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref ENEMY_AURA: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref PLAYER_BARRIER: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    pub static ref ENEMY_BARRIER: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    //xy of player grid position
    pub static ref PLAYER_GRID_POSITION: Arc<Mutex<(u16, u16)>> = Arc::new(Mutex::new((0, 0)));

    //xy of enemy grid position
    pub static ref ENEMY_GRID_POSITION: Arc<Mutex<(u16, u16)>> = Arc::new(Mutex::new((0, 0)));

    //is offerer
    pub static ref IS_OFFERER: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    //cust gage
    pub static ref CUST_GAGE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    // Global variable for storing player Navi Cust parts
    pub static ref PLAYER_NAVI_CUST_PARTS: Arc<Mutex<Option<Vec<usize>>>> = Arc::new(Mutex::new(None));

    // Global variable for storing enemy Navi Cust parts
    pub static ref ENEMY_NAVI_CUST_PARTS: Arc<Mutex<Option<Vec<usize>>>> = Arc::new(Mutex::new(None));

}


// ================== Time Frozen Getters and Setters ==================
pub fn set_time_frozen(value: u16) {
    let mut time_frozen = TIME_FROZEN.lock();
    *time_frozen = value;
}
pub fn get_time_frozen() -> u16 {
    let time_frozen = TIME_FROZEN.lock();
    *time_frozen
}

// ================== Player Aura Getters and Setters ==================
pub fn set_player_aura(value: u16) {
    let mut player_aura = PLAYER_AURA.lock();
    *player_aura = value;
}
pub fn get_player_aura() -> u16 {
    let player_aura = PLAYER_AURA.lock();
    *player_aura
}

// ================== Enemy Aura Getters and Setters ==================
pub fn set_enemy_aura(value: u16) {
    let mut enemy_aura = ENEMY_AURA.lock();
    *enemy_aura = value;
}
pub fn get_enemy_aura() -> u16 {
    let enemy_aura = ENEMY_AURA.lock();
    *enemy_aura
}

// ================== Player Barrier Getters and Setters ==================
pub fn set_player_barrier(value: u16) {
    let mut player_barrier = PLAYER_BARRIER.lock();
    *player_barrier = value;
}
pub fn get_player_barrier() -> u16 {
    let player_barrier = PLAYER_BARRIER.lock();
    *player_barrier
}

// ================== Enemy Barrier Getters and Setters ==================
pub fn set_enemy_barrier(value: u16) {
    let mut enemy_barrier = ENEMY_BARRIER.lock();
    *enemy_barrier = value;
}

pub fn get_enemy_barrier() -> u16 {
    let enemy_barrier = ENEMY_BARRIER.lock();
    *enemy_barrier
}

// ================== Grid Occupy State Getters and Setters ==================
pub fn set_grid_occupy_state(index: usize, value: u16) -> Result<(), String> {
    if index < GRID_OCCUPY_STATE.len() {
        let state = &GRID_OCCUPY_STATE[index];
        let mut state_lock = state.lock();
        *state_lock = value;
        Ok(())
    } else {
        Err(format!("Grid occupy state index {} out of bounds", index))
    }
}
pub fn get_grid_occupy_state(index: usize) -> Option<u16> {
    if index < GRID_OCCUPY_STATE.len() {
        let state = &GRID_OCCUPY_STATE[index];
        Some(*state.lock())
    } else {
        None
    }
}

pub fn get_all_grid_occupy_states() -> Vec<u16> {
    GRID_OCCUPY_STATE.iter().map(|state| *state.lock()).collect()
}

// ================== Player Invis Getters and Setters ==================
pub fn set_player_invis(value: u16) {
    let mut player_invis = PLAYER_INVIS.lock();
    *player_invis = value;
}
pub fn get_player_invis() -> u16 {
    let player_invis = PLAYER_INVIS.lock();
    *player_invis
}

// ================== Enemy Invis Getters and Setters ==================
pub fn set_enemy_invis(value: u16) {
    let mut enemy_invis = ENEMY_INVIS.lock();
    *enemy_invis = value;
}
pub fn get_enemy_invis() -> u16 {
    let enemy_invis = ENEMY_INVIS.lock();
    *enemy_invis
}

// ================== Player Trap Getters and Setters ==================
pub fn set_player_trap(value: u16) {
    let mut player_trap = PLAYER_TRAP.lock();
    *player_trap = value;
}

pub fn get_player_trap() -> u16 {
    let player_trap = PLAYER_TRAP.lock();
    *player_trap
}

// ================== Enemy Trap Getters and Setters ==================

pub fn set_enemy_trap(value: u16) {
    let mut enemy_trap = ENEMY_TRAP.lock();
    *enemy_trap = value;
}

pub fn get_enemy_trap() -> u16 {
    let enemy_trap = ENEMY_TRAP.lock();
    *enemy_trap
}

// ================== Player Navi Cust Parts Getters and Setters ==================
/// Sets the player Navi Cust parts
pub fn set_player_navi_cust_parts(parts: Vec<usize>) {
    let mut player_parts = PLAYER_NAVI_CUST_PARTS.lock();
    *player_parts = Some(parts);
}

/// Sets the enemy Navi Cust parts
pub fn set_enemy_navi_cust_parts(parts: Vec<usize>) {
    let mut enemy_parts = ENEMY_NAVI_CUST_PARTS.lock();
    *enemy_parts = Some(parts);
}

pub fn get_player_navi_cust_parts() -> Option<Vec<usize>> {
    let player_parts = PLAYER_NAVI_CUST_PARTS.lock();
    player_parts.clone()
}

pub fn get_enemy_navi_cust_parts() -> Option<Vec<usize>> {
    let enemy_parts = ENEMY_NAVI_CUST_PARTS.lock();
    enemy_parts.clone()
}


// ================== Cust Gage Getters and Setters ==================
pub fn set_cust_gage(value: u16) {
    let mut cust_gage = CUST_GAGE.lock();
    *cust_gage = value;
}
pub fn get_cust_gage() -> u16 {
    let cust_gage = CUST_GAGE.lock();
    *cust_gage
}

// ================== Is Offerer Getters and Setters ==================
pub fn set_is_offerer(value: u16) {
    let mut is_offerer = IS_OFFERER.lock();
    *is_offerer = value;
}
pub fn get_is_offerer() -> u16 {
    let is_offerer = IS_OFFERER.lock();
    *is_offerer
}

// ================== Grid Owner State Getters and Setters ==================
pub fn set_grid_owner_state(index: usize, value: u16) -> Result<(), String> {
    if index < GRID_OWNER_STATE.len() {
        let state = &GRID_OWNER_STATE[index];
        let mut state_lock = state.lock();
        *state_lock = value;
        Ok(())
    } else {
        Err(format!("Grid owner state index {} out of bounds", index))
    }
}
pub fn get_grid_owner_state(index: usize) -> Option<u16> {
    if index < GRID_OWNER_STATE.len() {
        let state = &GRID_OWNER_STATE[index];
        Some(*state.lock())
    } else {
        None
    }
}
pub fn get_all_grid_owner_states() -> Vec<u16> {
    GRID_OWNER_STATE.iter().map(|state| *state.lock()).collect()
}

// ================== Player Grid Position Getters and Setters ==================
pub fn set_player_grid_position(position: (u16, u16)) {
    let mut pos = PLAYER_GRID_POSITION.lock();
    *pos = position;
}
pub fn get_player_grid_position() -> Vec<u16> {
    let pos = PLAYER_GRID_POSITION.lock();
    vec![pos.0, pos.1]
}

// ================== Enemy Grid Position Getters and Setters ==================
pub fn set_enemy_grid_position(position: (u16, u16)) {
    let mut pos = ENEMY_GRID_POSITION.lock();
    *pos = position;
}
pub fn get_enemy_grid_position() -> Vec<u16> {
    let pos = ENEMY_GRID_POSITION.lock();
    vec![pos.0, pos.1]
}

// ================== Grid State Getters and Setters ==================
pub fn set_grid_state(index: usize, value: u16) -> Result<(), String> {
    if index < GRID_STATE.len() {
        let state = &GRID_STATE[index];
        let mut state_lock = state.lock();
        *state_lock = value;
        Ok(())
    } else {
        Err(format!("Grid state index {} out of bounds", index))
    }
}
pub fn get_grid_state(index: usize) -> Option<u16> {
    if index < GRID_STATE.len() {
        let state = &GRID_STATE[index];
        Some(*state.lock())
    } else {
        None
    }
}
pub fn get_all_grid_states() -> Vec<u16> {
    GRID_STATE.iter().map(|state| *state.lock()).collect()
}


// ================== Player Tag Folder Getters and Setters ==================
pub fn set_player_tag_folder(index: usize, value: u16) -> Result<(), String> {
    if index < PLAYER_TAG_FOLDER.len() {
        let folder = &PLAYER_TAG_FOLDER[index];
        let mut folder_lock = folder.lock();
        *folder_lock = value;
        Ok(())
    } else {
        Err(format!("Player tag folder index {} out of bounds", index))
    }
}
pub fn get_player_tag_folder(index: usize) -> Option<u16> {
    if index < PLAYER_TAG_FOLDER.len() {
        let folder = &PLAYER_TAG_FOLDER[index];
        Some(*folder.lock())
    } else {
        None
    }
}
pub fn get_all_player_tag_folders() -> Vec<u16> {
    PLAYER_TAG_FOLDER.iter().map(|folder| *folder.lock()).collect()
}

// ================== Enemy Tag Folder Getters and Setters ==================
pub fn set_enemy_tag_folder(index: usize, value: u16) -> Result<(), String> {
    if index < ENEMY_TAG_FOLDER.len() {
        let folder = &ENEMY_TAG_FOLDER[index];
        let mut folder_lock = folder.lock();
        *folder_lock = value;
        Ok(())
    } else {
        Err(format!("Enemy tag folder index {} out of bounds", index))
    }
}
pub fn get_enemy_tag_folder(index: usize) -> Option<u16> {
    if index < ENEMY_TAG_FOLDER.len() {
        let folder = &ENEMY_TAG_FOLDER[index];
        Some(*folder.lock())
    } else {
        None
    }
}
pub fn get_all_enemy_tag_folders() -> Vec<u16> {
    ENEMY_TAG_FOLDER.iter().map(|folder| *folder.lock()).collect()
}

// ================== Player Regular Chip Getters and Setters ==================
pub fn set_player_reg_chip(value: u16) {
    let mut reg_chip = PLAYER_REG_CHIP.lock();
    *reg_chip = value;
}
pub fn get_player_reg_chip() -> u16 {
    let reg_chip = PLAYER_REG_CHIP.lock();
    *reg_chip
}

// ================== Enemy Regular Chip Getters and Setters ==================
pub fn set_enemy_reg_chip(value: u16) {
    let mut reg_chip = ENEMY_REG_CHIP.lock();
    *reg_chip = value;
}

pub fn get_enemy_reg_chip() -> u16 {
    let reg_chip = ENEMY_REG_CHIP.lock();
    *reg_chip
}


// ================== Player Chip Folder Getters and Setters ==================
pub fn set_player_chip_folder(index: usize, value: u16) -> Result<(), String> {
    if index < PLAYER_CHIP_FOLDER.len() {
        let folder = &PLAYER_CHIP_FOLDER[index];
        let mut folder_lock = folder.lock();
        *folder_lock = value;
        Ok(())
    } else {
        Err(format!("Player chip folder index {} out of bounds", index))
    }
}
pub fn get_player_chip_folder(index: usize) -> Option<u16> {
    if index < PLAYER_CHIP_FOLDER.len() {
        let folder = &PLAYER_CHIP_FOLDER[index];
        Some(*folder.lock())
    } else {
        None
    }
}
pub fn get_all_player_chip_folders() -> Vec<u16> {
    PLAYER_CHIP_FOLDER.iter().map(|folder| *folder.lock()).collect()
}

// ================== Player Code Folder Getters and Setters ==================
pub fn set_player_code_folder(index: usize, value: u16) -> Result<(), String> {
    if index < PLAYER_CODE_FOLDER.len() {
        let folder = &PLAYER_CODE_FOLDER[index];
        let mut folder_lock = folder.lock();
        *folder_lock = value;
        Ok(())
    } else {
        Err(format!("Player code folder index {} out of bounds", index))
    }
}
pub fn get_player_code_folder(index: usize) -> Option<u16> {
    if index < PLAYER_CODE_FOLDER.len() {
        let folder = &PLAYER_CODE_FOLDER[index];
        Some(*folder.lock())
    } else {
        None
    }
}
pub fn get_all_player_code_folders() -> Vec<u16> {
    PLAYER_CODE_FOLDER.iter().map(|folder| *folder.lock()).collect()
}

// ================== Enemy Chip Folder Getters and Setters ==================
pub fn set_enemy_chip_folder(index: usize, value: u16) -> Result<(), String> {
    if index < ENEMY_CHIP_FOLDER.len() {
        let folder = &ENEMY_CHIP_FOLDER[index];
        let mut folder_lock = folder.lock();
        *folder_lock = value;
        Ok(())
    } else {
        Err(format!("Enemy chip folder index {} out of bounds", index))
    }
}

pub fn get_enemy_chip_folder(index: usize) -> Option<u16> {
    if index < ENEMY_CHIP_FOLDER.len() {
        let folder = &ENEMY_CHIP_FOLDER[index];
        Some(*folder.lock())
    } else {
        None
    }
}
pub fn get_all_enemy_chip_folders() -> Vec<u16> {
    ENEMY_CHIP_FOLDER.iter().map(|folder| *folder.lock()).collect()
}

// ================== Enemy Code Folder Getters and Setters ==================
pub fn set_enemy_code_folder(index: usize, value: u16) -> Result<(), String> {
    if index < ENEMY_CODE_FOLDER.len() {
        let folder = &ENEMY_CODE_FOLDER[index];
        let mut folder_lock = folder.lock();
        *folder_lock = value;
        Ok(())
    } else {
        Err(format!("Enemy code folder index {} out of bounds", index))
    }
}
pub fn get_enemy_code_folder(index: usize) -> Option<u16> {
    if index < ENEMY_CODE_FOLDER.len() {
        let folder = &ENEMY_CODE_FOLDER[index];
        Some(*folder.lock())
    } else {
        None
    }
}
pub fn get_all_enemy_code_folders() -> Vec<u16> {
    ENEMY_CODE_FOLDER.iter().map(|folder| *folder.lock()).collect()
}

// ================== Inside Cross Window Getters and Setters ==================
/// Sets the inside cross window flag
pub fn set_inside_cross_window(flag: u16) {
    let mut inside_cross_window = INSIDE_CROSS_WINDOW.lock();
    *inside_cross_window = flag;
}
pub fn get_inside_cross_window() -> u16 {
    let inside_cross_window = INSIDE_CROSS_WINDOW.lock();
    *inside_cross_window
}

// ================== Beast Out Selectable Getters and Setters ==================
/// Sets the beast out selectable flag
/// This flag is used to determine if the player can beast out
pub fn set_beast_out_selectable(selectable: u16) {
    let mut flag = BEAST_OUT_SELECTABLE.lock();
    *flag = selectable;
}
pub fn get_beast_out_selectable() -> u16 {
    let flag = BEAST_OUT_SELECTABLE.lock();
    *flag
}

// ================== RewardPunishment Getters and Setters ==================
/// Adds a reward to the global REWARDS list
pub fn add_reward(reward: RewardPunishment) {
    let mut rewards = REWARDS.lock();
    rewards.push(reward);
}

/// Adds a punishment to the global PUNISHMENTS list
pub fn add_punishment(punishment: RewardPunishment) {
    let mut punishments = PUNISHMENTS.lock();
    punishments.push(punishment);
}

/// Retrieves all rewards from the global REWARDS list
pub fn get_rewards() -> Vec<RewardPunishment> {
    let rewards = REWARDS.lock();
    rewards.clone()
}

/// Retrieves all punishments from the global PUNISHMENTS list
pub fn get_punishments() -> Vec<RewardPunishment> {
    let punishments = PUNISHMENTS.lock();
    punishments.clone()
}

/// Clears all rewards from the global REWARDS list
pub fn clear_rewards() {
    let mut rewards = REWARDS.lock();
    rewards.clear();
}

/// Clears all punishments from the global PUNISHMENTS list
pub fn clear_punishments() {
    let mut punishments = PUNISHMENTS.lock();
    punishments.clear();
}

// ================== Screen Image Getters and Setters ==================
/// Updates the global screen image with new pixel data
pub fn set_screen_image(image: ColorImage) {
    let mut screen_image = SCREEN_IMAGE.lock();
    *screen_image = Some(image);
}

/// Retrieves the current screen image
pub fn get_screen_image() -> Option<ColorImage> {
    let screen_image = SCREEN_IMAGE.lock();
    screen_image.clone()
}

// ================== Local Input Getters and Setters ==================
/// Sets the latest local input
pub fn set_local_input(input: u16) {
    let mut local_input = LOCAL_INPUT.lock();
    *local_input = Some(input);
}

/// Retrieves the latest local input
pub fn get_local_input() -> Option<u16> {
    let local_input = LOCAL_INPUT.lock();
    *local_input
}

/// Clears the latest local input after processing
pub fn clear_local_input() {
    let mut local_input = LOCAL_INPUT.lock();
    *local_input = None;
}

// ================== Replay Path Getters and Setters ==================
/// Sets the replay path
pub fn set_replay_path(path: String) {
    let mut replay_path = REPLAY_PATH.lock();
    *replay_path = Some(path);
}

/// Retrieves the replay path
pub fn get_replay_path() -> Option<String> {
    let replay_path = REPLAY_PATH.lock();
    replay_path.clone()
}

// ================== Winner State Getters and Setters ==================
/// Sets the winner state
pub fn set_winner(player_won: bool) {
    let mut winner_state = WINNER_STATE.lock();
    winner_state.player_won = Some(player_won);
}

/// Retrieves the winner state
pub fn get_winner() -> Option<bool> {
    let winner_state = WINNER_STATE.lock();
    winner_state.player_won
}

// ================== Player and Enemy Position Getters and Setters ==================
/// Sets the player position
pub fn set_player_position(position: (u16, u16)) {
    let mut pos = PLAYER_POSITION.lock();
    *pos = Some(position);
}

/// Retrieves the player position
pub fn get_player_position() -> Option<(u16, u16)> {
    let pos = PLAYER_POSITION.lock();
    *pos
}

/// Sets the enemy position
pub fn set_enemy_position(position: (u16, u16)) {
    let mut pos = ENEMY_POSITION.lock();
    *pos = Some(position);
}

/// Retrieves the enemy position
pub fn get_enemy_position() -> Option<(u16, u16)> {
    let pos = ENEMY_POSITION.lock();
    *pos
}

// ================== Player Forms Used Getters and Setters ==================
/// Sets the player forms used
pub fn set_player_forms_used(forms: Vec<String>) {
    let mut pf = PLAYER_FORMS_USED.lock();
    *pf = Some(forms);
}

/// Retrieves the player forms used
pub fn get_player_forms_used() -> Option<Vec<String>> {
    let pf = PLAYER_FORMS_USED.lock();
    pf.clone()
}

// ================== Enemy Forms Used Getters and Setters ==================
/// Sets the enemy forms used
pub fn set_enemy_forms_used(forms: Vec<String>) {
    let mut ef = ENEMY_FORMS_USED.lock();
    *ef = Some(forms);
}

/// Retrieves the enemy forms used
pub fn get_enemy_forms_used() -> Option<Vec<String>> {
    let ef = ENEMY_FORMS_USED.lock();
    ef.clone()
}

// ================== Is Player Inside Window Getters and Setters ==================
/// Sets the flag indicating if the player is inside the window
pub fn set_is_player_inside_window(is_inside: bool) {
    let mut flag = IS_PLAYER_INSIDE_WINDOW.lock();
    *flag = Some(is_inside);
}

/// Retrieves the flag indicating if the player is inside the window
pub fn get_is_player_inside_window() -> Option<bool> {
    let flag = IS_PLAYER_INSIDE_WINDOW.lock();
    *flag
}

// ================== Player and Enemy Health Getters and Setters ==================
/// Sets the player health
pub fn set_player_health(health: u16) {
    let mut player_health = PLAYER_HEALTH.lock();
    *player_health = health;
}

/// Retrieves the player health
pub fn get_player_health() -> u16 {
    let player_health = PLAYER_HEALTH.lock();
    *player_health
}

/// Sets the enemy health
pub fn set_enemy_health(health: u16) {
    let mut enemy_health = ENEMY_HEALTH.lock();
    *enemy_health = health;
}

/// Retrieves the enemy health
pub fn get_enemy_health() -> u16 {
    let enemy_health = ENEMY_HEALTH.lock();
    *enemy_health
}

// ================== Player Health Index Getters and Setters ==================
/// Sets the player health index
pub fn set_player_health_index(index: u16) {
    println!("Setting player health index to: {:?}", index);
    let mut player_health_index = PLAYER_HEALTH_INDEX.lock();
    *player_health_index = index;
}

/// Retrieves the player health index
pub fn get_player_health_index() -> u16 {
    let player_health_index = PLAYER_HEALTH_INDEX.lock();
    *player_health_index
}

// ================== Frame Count Getters and Setters ==================
/// Sets the frame count
pub fn set_frame_count(count: u16) {
    let mut frame_count = FRAME_COUNT.lock();
    *frame_count = count;
}

/// Retrieves the frame count
pub fn get_frame_count() -> u16 {
    let frame_count = FRAME_COUNT.lock();
    *frame_count
}

// ================== Player Charge Getters and Setters ==================
/// Sets the player charge
pub fn set_player_charge(charge: u16) {
    let mut player_charge = PLAYER_CHARGE.lock();
    *player_charge = charge;
}

/// Retrieves the player charge
pub fn get_player_charge() -> u16 {
    let player_charge = PLAYER_CHARGE.lock();
    *player_charge
}

// ================== Enemy Charge Getters and Setters ==================
/// Sets the enemy charge
pub fn set_enemy_charge(charge: u16) {
    let mut enemy_charge = ENEMY_CHARGE.lock();
    *enemy_charge = charge;
}

/// Retrieves the enemy charge
pub fn get_enemy_charge() -> u16 {
    let enemy_charge = ENEMY_CHARGE.lock();
    *enemy_charge
}

// ================== Player Selected Chip Getters and Setters ==================
/// Sets the player selected chip
pub fn set_player_selected_chip(chip: u16) {
    let mut player_selected_chip = PLAYER_SELECTED_CHIP.lock();
    *player_selected_chip = chip;
}

/// Retrieves the player selected chip
pub fn get_player_selected_chip() -> u16 {
    let player_selected_chip = PLAYER_SELECTED_CHIP.lock();
    *player_selected_chip
}

// ================== Enemy Selected Chip Getters and Setters ==================
/// Sets the enemy selected chip
pub fn set_enemy_selected_chip(chip: u16) {
    let mut enemy_selected_chip = ENEMY_SELECTED_CHIP.lock();
    *enemy_selected_chip = chip;
}

/// Retrieves the enemy selected chip
pub fn get_enemy_selected_chip() -> u16 {
    let enemy_selected_chip = ENEMY_SELECTED_CHIP.lock();
    *enemy_selected_chip
}

// ================== Player Emotion State Getters and Setters ==================
/// Sets the player emotion state
pub fn set_player_emotion_state(state: u16) {
    let mut player_emotion_state = PLAYER_EMOTION_STATE.lock();
    *player_emotion_state = state;
}

/// Retrieves the player emotion state
pub fn get_player_emotion_state() -> u16 {
    let player_emotion_state = PLAYER_EMOTION_STATE.lock();
    *player_emotion_state
}

// ================== Enemy Emotion State Getters and Setters ==================
/// Sets the enemy emotion state
pub fn set_enemy_emotion_state(state: u16) {
    let mut enemy_emotion_state = ENEMY_EMOTION_STATE.lock();
    *enemy_emotion_state = state;
}

/// Retrieves the enemy emotion state
pub fn get_enemy_emotion_state() -> u16 {
    let enemy_emotion_state = ENEMY_EMOTION_STATE.lock();
    *enemy_emotion_state
}

// ================== Player Game Emotion State Getters and Setters ==================
/// Sets the player game emotion state
/// This is the emotion state that is used in the game
/// It is different from the player emotion state
pub fn set_player_game_emotion_state(state: u16) {
    let mut player_game_emotion_state = PLAYER_GAME_EMOTION_STATE.lock();
    *player_game_emotion_state = state;
}
pub fn set_enemy_game_emotion_state(state: u16) {
    let mut enemy_game_emotion_state = ENEMY_GAME_EMOTION_STATE.lock();
    *enemy_game_emotion_state = state;
}

/// Retrieves the player game emotion state
pub fn get_player_game_emotion_state() -> u16 {
    let player_game_emotion_state = PLAYER_GAME_EMOTION_STATE.lock();
    *player_game_emotion_state
}
pub fn get_enemy_game_emotion_state() -> u16 {
    let enemy_game_emotion_state = ENEMY_GAME_EMOTION_STATE.lock();
    *enemy_game_emotion_state
}

// ================== Selected Menu Index Getters and Setters ==================
/// Sets the selected menu index
pub fn set_selected_menu_index(index: u16) {
    let mut selected_menu_index = SELECTED_MENU_INDEX.lock();
    *selected_menu_index = index;
}

/// Retrieves the selected menu index
pub fn get_selected_menu_index() -> u16 {
    let selected_menu_index = SELECTED_MENU_INDEX.lock();
    *selected_menu_index
}

// ================== Selected Cross Index Getters and Setters ==================
/// Sets the selected cross index
pub fn set_selected_cross_index(index: u16) {
    let mut selected_cross_index = SELECTED_CROSS_INDEX.lock();
    *selected_cross_index = index;
}

/// Retrieves the selected cross index
pub fn get_selected_cross_index() -> u16 {
    let selected_cross_index = SELECTED_CROSS_INDEX.lock();
    *selected_cross_index
}

// ================== Chip Selected Count Getters and Setters ==================
/// Sets the chip selected count
pub fn set_chip_selected_count(count: u16) {
    let mut chip_selected_count = CHIP_SELECTED_COUNT.lock();
    *chip_selected_count = count;
}

/// Retrieves the chip selected count
pub fn get_chip_selected_count() -> u16 {
    let chip_selected_count = CHIP_SELECTED_COUNT.lock();
    *chip_selected_count
}

// ================== Chip Count Visible Getters and Setters ==================
/// Sets the chip count visible
pub fn set_chip_count_visible(count: u16) {
    let mut chip_count_visible = CHIP_COUNT_VISIBLE.lock();
    *chip_count_visible = count;
}
pub fn get_chip_count_visible() -> u16 {
    let chip_count_visible = CHIP_COUNT_VISIBLE.lock();
    *chip_count_visible
}

// ================== Chip Slots Getters and Setters ==================
/// Sets a specific chip slot by index
pub fn set_chip_slot(index: usize, value: u16) -> Result<(), String> {
    if index < CHIP_SLOTS.len() {
        let slot = &CHIP_SLOTS[index];
        let mut slot_lock = slot.lock();
        *slot_lock = value;
        Ok(())
    } else {
        Err(format!("Chip slot index {} out of bounds", index))
    }
}

/// Retrieves a specific chip slot by index
pub fn get_chip_slot(index: usize) -> Option<u16> {
    if index < CHIP_SLOTS.len() {
        let slot = &CHIP_SLOTS[index];
        Some(*slot.lock())
    } else {
        None
    }
}


/// Retrieves all chip slots
pub fn get_all_chip_slots() -> Vec<u16> {
    CHIP_SLOTS.iter().map(|slot| *slot.lock()).collect()
}

// ================== Chip Codes Getters and Setters ==================
/// Sets a specific chip code by index
pub fn set_chip_code(index: usize, value: u16) -> Result<(), String> {
    if index < CHIP_CODES.len() {
        let code = &CHIP_CODES[index];
        let mut code_lock = code.lock();
        *code_lock = value;
        Ok(())
    } else {
        Err(format!("Chip code index {} out of bounds", index))
    }
}

/// Retrieves a specific chip code by index
pub fn get_chip_code(index: usize) -> Option<u16> {
    if index < CHIP_CODES.len() {
        let code = &CHIP_CODES[index];
        Some(*code.lock())
    } else {
        None
    }
}

/// Retrieves all chip codes
pub fn get_all_chip_codes() -> Vec<u16> {
    CHIP_CODES.iter().map(|code| *code.lock()).collect()
}

// ================== Selected Chip Indices Getters and Setters ==================
/// Sets a specific selected chip index by index
pub fn set_selected_chip_index(index: usize, value: u16) -> Result<(), String> {
    if index < SELECTED_CHIP_INDICES.len() {
        let selected_index = &SELECTED_CHIP_INDICES[index];
        let mut selected_lock = selected_index.lock();
        *selected_lock = value;
        Ok(())
    } else {
        Err(format!("Selected chip index {} out of bounds", index))
    }
}

/// Retrieves a specific selected chip index by index
pub fn get_selected_chip_index(index: usize) -> Option<u16> {
    if index < SELECTED_CHIP_INDICES.len() {
        let selected_index = &SELECTED_CHIP_INDICES[index];
        Some(*selected_index.lock())
    } else {
        None
    }
}

/// Retrieves all selected chip indices
pub fn get_all_selected_chip_indices() -> Vec<u16> {
    SELECTED_CHIP_INDICES.iter().map(|idx| *idx.lock()).collect()
}
