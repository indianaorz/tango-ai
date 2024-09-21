// src/global.rs

use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Arc;
use egui::ColorImage;
use serde::Serialize;
// Assuming RewardPunishment is defined like this:
#[derive(Debug, Clone,Serialize)]
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

    //player and enemy health
    pub static ref PLAYER_HEALTH: Arc<Mutex<u16>> = Arc::new(Mutex::new(100));
    pub static ref ENEMY_HEALTH: Arc<Mutex<u16>> = Arc::new(Mutex::new(100));

    //player health index
    pub static ref PLAYER_HEALTH_INDEX: Arc<Mutex<u16>> = Arc::new(Mutex::new(2));

    //frame count
    pub static ref FRAME_COUNT: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    //player charge
    pub static ref PLAYER_CHARGE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));

    //enemy charge
    pub static ref ENEMY_CHARGE: Arc<Mutex<u16>> = Arc::new(Mutex::new(0));
}


// Function to set the player charge
pub fn set_player_charge(charge: u16) {
    let mut player_charge = PLAYER_CHARGE.lock();
    *player_charge = charge;
}

pub fn get_player_charge() -> u16 {
    let player_charge = PLAYER_CHARGE.lock();
    *player_charge
}

// Function to set the enemy charge
pub fn set_enemy_charge(charge: u16) {
    let mut enemy_charge = ENEMY_CHARGE.lock();
    *enemy_charge = charge;
}

pub fn get_enemy_charge() -> u16 {
    let enemy_charge = ENEMY_CHARGE.lock();
    *enemy_charge
}

// Function to set the frame count
pub fn set_frame_count(count: u16) {
    let mut frame_count = FRAME_COUNT.lock();
    *frame_count = count;
}

pub fn get_frame_count() -> u16 {
    let frame_count = FRAME_COUNT.lock();
    *frame_count
}

// Function to set the player health index
pub fn set_player_health_index(index: u16) {
    println!("Setting player health index to: {:?}", index);
    let mut player_health_index = PLAYER_HEALTH_INDEX.lock();
    *player_health_index = index;
}
pub fn get_player_health_index() -> u16 {
    let player_health_index = PLAYER_HEALTH_INDEX.lock();
    *player_health_index
}

// Function to set the player health
pub fn set_player_health(health: u16) {
    let mut player_health = PLAYER_HEALTH.lock();
    *player_health = health;
}
pub fn get_player_health() -> u16 {
    let player_health = PLAYER_HEALTH.lock();
    *player_health
}
//enemy health
pub fn set_enemy_health(health: u16) {
    let mut enemy_health = ENEMY_HEALTH.lock();
    *enemy_health = health;
}
pub fn get_enemy_health() -> u16 {
    let enemy_health = ENEMY_HEALTH.lock();
    *enemy_health
}

// Function to set the winner state
pub fn set_winner(player_won: bool) {
    let mut winner_state = WINNER_STATE.lock();
    winner_state.player_won = Some(player_won);
}

// Function to get the winner state
pub fn get_winner() -> Option<bool> {
    let winner_state = WINNER_STATE.lock();
    winner_state.player_won
}

// Static methods for adding and clearing rewards and punishments

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



// Function to set the latest input
pub fn set_local_input(input: u16) {
    let mut local_input = LOCAL_INPUT.lock();
    *local_input = Some(input);
    // println!("Set local input: {:?}", input);
}

// Function to retrieve the latest input
pub fn get_local_input() -> Option<u16> {
    let local_input = LOCAL_INPUT.lock();
    local_input.clone()
}

// Function to clear the latest input after processing
pub fn clear_local_input() {
    let mut local_input = LOCAL_INPUT.lock();
    *local_input = None;
    // println!("Cleared local input");
}


// Function to set the replay path
pub fn set_replay_path(path: String) {
    let mut replay_path = REPLAY_PATH.lock();
    *replay_path = Some(path);
}

// Function to get the replay path
pub fn get_replay_path() -> Option<String> {
    let replay_path = REPLAY_PATH.lock();
    replay_path.clone()
}



// Player Position
pub fn set_player_position(position: (u16, u16)) {
    let mut pos = PLAYER_POSITION.lock();
    *pos = Some(position);
}

pub fn get_player_position() -> Option<(u16, u16)> {
    let pos = PLAYER_POSITION.lock();
    *pos
}

// Enemy Position
pub fn set_enemy_position(position: (u16, u16)) {
    let mut pos = ENEMY_POSITION.lock();
    *pos = Some(position);
}

pub fn get_enemy_position() -> Option<(u16, u16)> {
    let pos = ENEMY_POSITION.lock();
    *pos
}

// Grid State
pub fn set_grid_state(state: Vec<String>) {
    let mut gs = GRID_STATE.lock();
    *gs = Some(state);
}

pub fn get_grid_state() -> Option<Vec<String>> {
    let gs = GRID_STATE.lock();
    gs.clone()
}

// Player Forms Used
pub fn set_player_forms_used(forms: Vec<String>) {
    let mut pf = PLAYER_FORMS_USED.lock();
    *pf = Some(forms);
}

pub fn get_player_forms_used() -> Option<Vec<String>> {
    let pf = PLAYER_FORMS_USED.lock();
    pf.clone()
}

// Enemy Forms Used
pub fn set_enemy_forms_used(forms: Vec<String>) {
    let mut ef = ENEMY_FORMS_USED.lock();
    *ef = Some(forms);
}

pub fn get_enemy_forms_used() -> Option<Vec<String>> {
    let ef = ENEMY_FORMS_USED.lock();
    ef.clone()
}

// Is Player Inside Window
pub fn set_is_player_inside_window(is_inside: bool) {
    let mut flag = IS_PLAYER_INSIDE_WINDOW.lock();
    *flag = Some(is_inside);
}

pub fn get_is_player_inside_window() -> Option<bool> {
    let flag = IS_PLAYER_INSIDE_WINDOW.lock();
    *flag
}