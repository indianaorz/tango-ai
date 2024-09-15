// src/global.rs

use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::sync::Arc;
use egui::ColorImage;

// Assuming RewardPunishment is defined like this:
#[derive(Debug, Clone)]
pub struct RewardPunishment {
    pub damage: u16,
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