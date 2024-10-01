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
    pub static ref GRID_STATE: Arc<Mutex<Option<Vec<String>>>> = Arc::new(Mutex::new(None));
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

    // Chip slots
    pub static ref CHIP_SLOTS: Vec<Arc<Mutex<u16>>> = (0..8).map(|_| Arc::new(Mutex::new(0))).collect();

    // Chip codes
    pub static ref CHIP_CODES: Vec<Arc<Mutex<u16>>> = (0..8).map(|_| Arc::new(Mutex::new(0))).collect();

    // 5 possible ids for selected chips
    pub static ref SELECTED_CHIP_INDICES: Vec<Arc<Mutex<u16>>> = (0..5).map(|_| Arc::new(Mutex::new(0))).collect();
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

// ================== Grid State Getters and Setters ==================
/// Sets the grid state
pub fn set_grid_state(state: Vec<String>) {
    let mut gs = GRID_STATE.lock();
    *gs = Some(state);
}

/// Retrieves the grid state
pub fn get_grid_state() -> Option<Vec<String>> {
    let gs = GRID_STATE.lock();
    gs.clone()
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
