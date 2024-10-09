use crate::global::{
    set_beast_out_selectable, set_chip_code, set_chip_count_visible, set_chip_selected_count, set_chip_slot, set_cust_gage, set_enemy_game_emotion_state, set_enemy_trap, set_grid_occupy_state, set_inside_cross_window, set_is_offerer, set_player_barrier, set_player_chip_folder, set_player_emotion_state, set_player_game_emotion_state, set_player_invis, set_player_trap, set_selected_chip_index, set_selected_cross_index, set_selected_menu_index, set_time_frozen
};
use crate::{audio, config, game, gui, net, rom, stats, video};
use egui::debug_text::print;
use mgba::input::keys::L;
use parking_lot::Mutex;
use rand::SeedableRng;
use sha2::digest::consts::True;
use std::sync::Arc;

use std::collections::HashMap;
use std::io::Write;

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use std::fs::File;

use std::sync::LazyLock;

pub const EXPECTED_FPS: f32 = (16777216.0 / 280896.0) * 4.0;

// session.rs
use crate::global::{
    add_punishment, add_reward, clear_punishments, clear_rewards, get_frame_count, get_player_health_index,
    set_enemy_charge, set_enemy_chip_folder, set_enemy_code_folder, set_enemy_grid_position, set_enemy_health,
    set_enemy_navi_cust_parts, set_enemy_position, set_enemy_reg_chip, set_enemy_selected_chip, set_enemy_tag_folder,
    set_frame_count, set_grid_owner_state, set_grid_state, set_is_player_inside_window, set_local_input,
    set_player_charge, set_player_code_folder, set_player_grid_position, set_player_health, set_player_health_index,
    set_player_navi_cust_parts, set_player_position, set_player_reg_chip, set_player_selected_chip,set_player_aura,
    set_player_tag_folder, set_winner, RewardPunishment,
};
use crate::global::{PUNISHMENTS, REWARDS}; // Import the global variables

enum AddressSize {
    U8,
    U16,
    U32,
}

pub struct GameInfo {
    pub game: &'static (dyn game::Game + Send + Sync),
    pub patch: Option<(String, semver::Version)>,
}

pub struct Setup {
    pub game_lang: unic_langid::LanguageIdentifier,
    pub save: Box<dyn tango_dataview::save::Save + Send + Sync>,
    pub assets: Box<dyn tango_dataview::rom::Assets + Send + Sync>,
}

pub struct Session {
    start_time: std::time::SystemTime,
    game_info: GameInfo,
    vbuf: std::sync::Arc<Mutex<Vec<u8>>>,
    _audio_binding: audio::Binding,
    thread: mgba::thread::Thread,
    joyflags: std::sync::Arc<std::sync::atomic::AtomicU32>,
    mode: Mode,
    completion_token: tango_pvp::hooks::CompletionToken,
    pause_on_next_frame: std::sync::Arc<std::sync::atomic::AtomicBool>,
    opponent_setup: Option<Setup>,
    own_setup: Option<Setup>,
}

pub struct PvP {
    pub match_: std::sync::Arc<tokio::sync::Mutex<Option<std::sync::Arc<tango_pvp::battle::Match>>>>,
    cancellation_token: tokio_util::sync::CancellationToken,
    latency_counter: std::sync::Arc<tokio::sync::Mutex<crate::stats::LatencyCounter>>,
    _peer_conn: datachannel_wrapper::PeerConnection,
}

impl PvP {
    pub async fn latency(&self) -> std::time::Duration {
        self.latency_counter.lock().await.median()
    }
}

pub struct SinglePlayer {}

pub enum Mode {
    SinglePlayer(SinglePlayer),
    PvP(PvP),
    Replayer,
}

use mgba::core::CoreMutRef;

use tango_dataview::rom::Assets;
use tango_dataview::rom::NavicustPart;
use tango_dataview::save::NaviView;
use tango_dataview::save::Save;

/// Extracts Navi Cust IDs from the save data.
///
/// # Arguments
///
/// * `save` - A reference to the game's save data.
/// * `assets` - A reference to the game's assets.
///
/// # Returns
///
/// A vector containing the IDs of all Navi Cust parts.
fn extract_navi_cust_ids(save: &dyn Save, assets: &dyn Assets) -> Vec<usize> {
    // Access the Navi Cust view from the save data
    if let Some(navi_view) = save.view_navi() {
        match navi_view {
            NaviView::Navicust(navicust_view) => {
                let size = navicust_view.size(); // Typically [7, 7]
                let mut navi_cust_ids = Vec::new();

                for i in 0..(size[0] * size[1]) {
                    if let Some(part) = navicust_view.navicust_part(i) {
                        navi_cust_ids.push(part.id);
                    }
                }

                navi_cust_ids
            }
            NaviView::LinkNavi(_) => {
                // Handle LinkNavi if necessary, or ignore
                Vec::new()
            }
        }
    } else {
        Vec::new()
    }
}

use std::error::Error;
use tango_dataview::save::Chip;

/// Extracts all chips from the equipped folder in the save data.
///
/// # Arguments
///
/// * `save` - A reference to the game's save data.
/// * `assets` - A reference to the game's assets.
///
/// # Returns
///
/// A vector containing all the chips from the equipped folder.
fn extract_chips(save: &dyn Save, assets: &dyn Assets) -> Result<Vec<Chip>, Box<dyn Error>> {
    let mut chips = Vec::new();

    if let Some(chips_view) = save.view_chips() {
        // Get the index of the currently equipped folder
        let equipped_folder_index = chips_view.equipped_folder_index();

        // Iterate through chip indices 0 to 29 in the equipped folder
        for chip_index in 0..30 {
            if let Some(chip) = chips_view.chip(equipped_folder_index, chip_index) {
                chips.push(chip);
            }
        }
    }

    Ok(chips)
}

/// Extracts the indices of the tag chips from the equipped folder in the save data.
///
/// # Arguments
///
/// * `save` - A reference to the game's save data.
/// * `assets` - A reference to the game's assets.
///
/// # Returns
///
/// A vector containing the indices of all tag chips in the equipped folder.
fn extract_tag_indices(save: &dyn Save, assets: &dyn Assets) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut tag_chip_indices = Vec::new();

    if let Some(chips_view) = save.view_chips() {
        // Get the index of the currently equipped folder
        let equipped_folder_index = chips_view.equipped_folder_index();

        // Retrieve the tag chip indices for the equipped folder
        if let Some(tag_indices) = chips_view.tag_chip_indexes(equipped_folder_index) {
            for &chip_index in &tag_indices {
                tag_chip_indices.push(chip_index);
            }
        }
    }

    Ok(tag_chip_indices)
}

fn extract_reg_index(save: &dyn Save, assets: &dyn Assets) -> Result<usize, Box<dyn Error>> {
    let mut reg_index = 0;

    if let Some(chips_view) = save.view_chips() {
        // Get the index of the currently equipped folder
        let equipped_folder_index = chips_view.equipped_folder_index();

        // Retrieve the regular chip index for the equipped folder
        if let Some(index) = chips_view.regular_chip_index(equipped_folder_index) {
            reg_index = index;
        }
    }

    Ok(reg_index)
}
use std::sync::atomic::AtomicU8;
impl Session {
    pub fn new_pvp(
        config: std::sync::Arc<parking_lot::RwLock<config::Config>>,
        audio_binder: audio::LateBinder,
        link_code: String,
        netplay_compatibility: String,
        local_settings: net::protocol::Settings,
        local_game: &'static (dyn game::Game + Send + Sync),
        local_patch: Option<(String, semver::Version)>,
        local_patch_overrides: &rom::Overrides,
        local_rom: &[u8],
        local_save: Box<dyn tango_dataview::save::Save + Send + Sync + 'static>,
        remote_settings: net::protocol::Settings,
        remote_game: &'static (dyn game::Game + Send + Sync),
        remote_patch_overrides: &rom::Overrides,
        remote_rom: &[u8],
        remote_save: Box<dyn tango_dataview::save::Save + Send + Sync + 'static>,
        emu_tps_counter: Arc<Mutex<stats::Counter>>,
        sender: net::Sender,
        receiver: net::Receiver,
        peer_conn: datachannel_wrapper::PeerConnection,
        is_offerer: bool,
        replays_path: std::path::PathBuf,
        match_type: (u8, u8),
        rng_seed: [u8; 16],
    ) -> Result<Self, anyhow::Error> {
        // Step 1: Initialize a shared variable for local_player_index
        let local_player_index_shared = Arc::new(AtomicU8::new(0));

        // Clone it to move into the closure
        let local_player_index_clone = Arc::clone(&local_player_index_shared);

        let mut core = mgba::core::Core::new_gba("tango")?;
        core.enable_video_buffer();

        core.as_mut()
            .load_rom(mgba::vfile::VFile::from_vec(local_rom.to_vec()))?;
        core.as_mut()
            .load_save(mgba::vfile::VFile::from_vec(local_save.as_sram_dump()))?;

        let joyflags = Arc::new(std::sync::atomic::AtomicU32::new(0));

        let local_hooks = tango_pvp::hooks::hooks_for_gamedb_entry(local_game.gamedb_entry()).unwrap();
        local_hooks.patch(core.as_mut());

        let match_ = std::sync::Arc::new(tokio::sync::Mutex::new(None));
        let _ = std::fs::create_dir_all(replays_path.parent().unwrap());
        let mut traps = local_hooks.common_traps();

        let completion_token = tango_pvp::hooks::CompletionToken::new();

        traps.extend(local_hooks.primary_traps(joyflags.clone(), match_.clone(), completion_token.clone()));

        core.set_traps(
            traps
                .into_iter()
                .map(|(addr, f)| {
                    let handle = tokio::runtime::Handle::current();
                    (
                        addr,
                        Box::new(move |core: mgba::core::CoreMutRef<'_>| {
                            let _guard = handle.enter();
                            f(core)
                        }) as Box<dyn Fn(mgba::core::CoreMutRef<'_>)>,
                    )
                })
                .collect(),
        );

        let reveal_setup = remote_settings.reveal_setup;

        let thread = mgba::thread::Thread::new(core);

        let sender = std::sync::Arc::new(tokio::sync::Mutex::new(sender));
        let latency_counter = std::sync::Arc::new(tokio::sync::Mutex::new(crate::stats::LatencyCounter::new(5)));

        // Create a shared variable for local_player_index
        let local_player_index_arc = Arc::new(AtomicU8::new(0));
        let local_player_index_for_match = Arc::clone(&local_player_index_arc);

        let cancellation_token = tokio_util::sync::CancellationToken::new();
        let match_ = match_.clone();
        *match_.try_lock().unwrap() = Some({
            let config = config.read();
            let replays_path = config.replays_path();
            let link_code = link_code.clone();
            let netplay_compatibility = netplay_compatibility.clone();
            let local_settings = local_settings.clone();
            let remote_settings = remote_settings.clone();
            let replaycollector_endpoint = config.replaycollector_endpoint.clone();
            let inner_match = tango_pvp::battle::Match::new(
                local_rom.to_vec(),
                local_hooks,
                tango_pvp::hooks::hooks_for_gamedb_entry(remote_game.gamedb_entry()).unwrap(),
                cancellation_token.clone(),
                Box::new(crate::net::PvpSender::new(sender.clone())),
                rand_pcg::Mcg128Xsl64::from_seed(rng_seed),
                is_offerer,
                thread.handle(),
                remote_rom,
                remote_save.as_ref(),
                match_type,
                config.input_delay,
                move |round_number, local_player_index| {
                    // Step 3: Update the shared local_player_index
                    local_player_index_for_match.store(local_player_index, Ordering::SeqCst);

                    const TIME_DESCRIPTION: &[time::format_description::FormatItem<'_>] = time::macros::format_description!(
                        "[year padding:zero][month padding:zero repr:numerical][day padding:zero][hour padding:zero][minute padding:zero][second padding:zero]"
                    );
                    //if round number is 2, then close the app
                    if round_number == 2 {
                        //close the program
                        // Exit the application after sending the winner message
                        std::process::exit(0);
                    }
                    let replay_filename = replays_path.join(format!(
                        "{}.tangoreplay",
                        format!(
                            "{}-{}-{}-vs-{}-round{}-p{}",
                            time::OffsetDateTime::from(std::time::SystemTime::now())
                                .format(TIME_DESCRIPTION)
                                .expect("format time"),
                            link_code,
                            netplay_compatibility,
                            remote_settings.nickname,
                            round_number,
                            local_player_index + 1
                        )
                        .chars()
                        .filter(|c| "/\\?%*:|\"<>. ".chars().all(|c2| c2 != *c))
                        .collect::<String>()
                    ));
                    log::info!("open replay: {}", replay_filename.display());

                    let local_game_settings = local_settings.game_info.as_ref().unwrap();
                    let remote_game_settings = remote_settings.game_info.as_ref().unwrap();

                    let replay_file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&replay_filename)?;
                    Ok(Some(tango_pvp::replay::Writer::new(
                        replay_file,
                        tango_pvp::replay::Metadata {
                            ts: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64,
                            link_code: link_code.clone(),
                            local_side: Some(tango_pvp::replay::metadata::Side {
                                nickname: local_settings.nickname.clone(),
                                game_info: Some(tango_pvp::replay::metadata::GameInfo {
                                    rom_family: local_game_settings.family_and_variant.0.to_string(),
                                    rom_variant: local_game_settings.family_and_variant.1 as u32,
                                    patch: local_game_settings.patch.as_ref().map(|patch|
                                        tango_pvp::replay::metadata::game_info::Patch {
                                            name: patch.name.clone(),
                                            version: patch.version.to_string(),
                                        }
                                    ),
                                }),
                                reveal_setup: local_settings.reveal_setup,
                            }),
                            remote_side: Some(tango_pvp::replay::metadata::Side {
                                nickname: remote_settings.nickname.clone(),
                                game_info: Some(tango_pvp::replay::metadata::GameInfo {
                                    rom_family: remote_game_settings.family_and_variant.0.to_string(),
                                    rom_variant: remote_game_settings.family_and_variant.1 as u32,
                                    patch: remote_game_settings.patch.as_ref().map(|patch|
                                        tango_pvp::replay::metadata::game_info::Patch {
                                            name: patch.name.clone(),
                                            version: patch.version.to_string(),
                                        }
                                    ),
                                }),
                                reveal_setup: remote_settings.reveal_setup,
                            }),
                            round: round_number as u32,
                            match_type: match_type.0 as u32,
                            match_subtype: match_type.1 as u32,
                        },
                        local_player_index,
                        local_hooks.packet_size() as u8,
                    )?))
                },
                move |r| {
                    if replaycollector_endpoint.is_empty() {
                        return Ok(());
                    }

                    let mut buf = vec![];
                    r.read_to_end(&mut buf)?;

                    let replaycollector_endpoint = replaycollector_endpoint.clone();

                    tokio::spawn(async move {
                        if let Err(e) = async move {
                            let client = reqwest::Client::new();
                            client
                                .post(replaycollector_endpoint)
                                .header("Content-Type", "application/x-tango-replay")
                                .body(buf)
                                .send()
                                .await?
                                .error_for_status()?;
                            Ok::<(), anyhow::Error>(())
                        }
                        .await
                        {
                            log::error!("failed to submit replay: {:?}", e);
                        }
                    });

                    Ok(())
                },
            )
            .expect("new match");

            {
                let match_ = match_.clone();
                let inner_match = inner_match.clone();
                let receiver = Box::new(crate::net::PvpReceiver::new(
                    receiver,
                    sender.clone(),
                    latency_counter.clone(),
                ));
                tokio::task::spawn(async move {
                    tokio::select! {
                        r = inner_match.run(receiver) => {
                            log::info!("match thread ending: {:?}", r);
                        }
                        _ = inner_match.cancelled() => {
                            //exit application
                            std::process::exit(0);
                        }
                    }
                    log::info!("match thread ended");
                    std::process::exit(0);
                    *match_.lock().await = None;
                });
            }

            inner_match
        });

        thread.start()?;
        thread.handle().lock_audio().sync_mut().set_fps_target(EXPECTED_FPS);

        let audio_binding = audio_binder.bind(Some(Box::new(audio::MGBAStream::new(
            thread.handle(),
            audio_binder.sample_rate(),
        ))))?;

        let vbuf = Arc::new(Mutex::new(vec![
            0u8;
            (mgba::gba::SCREEN_WIDTH * mgba::gba::SCREEN_HEIGHT * 4)
                as usize
        ]));

        fn search_all_health_values(core: &mut CoreMutRef) -> Vec<u32> {
            let segment = -1; // Default segment, adjust as necessary
            let value_to_find = 1800;
            let mut found_addresses = Vec::new();

            // Define the memory range to search - adjust based on your game's addressable space
            let start_address = 0x02000000; // Start of EWRAM, commonly used in GBA games
            let end_address = 0x02040000; // End of EWRAM

            // Search through the address range for the value 1800
            for address in (start_address..end_address).step_by(4) {
                let current_value = core.raw_read_16(address, segment);
                if current_value == value_to_find || current_value == 1400 {
                    found_addresses.push(address);
                }
            }

            found_addresses
        }

        //search all boolean values to find which ones are changing
        fn search_all_boolean_values(core: &mut CoreMutRef) -> Vec<u32> {
            let segment = -1; // Default segment, adjust as necessary
            let mut found_addresses = Vec::new();

            // Define the memory range to search - adjust based on your game's addressable space
            let start_address = 0x02000000; // Start of EWRAM, commonly used in GBA games
            let end_address = 0x02050000; // End of EWRAM

            // Search through the address range for the value 1800
            for address in (start_address..end_address).step_by(1) {
                let current_value = core.raw_read_8(address, segment);
                // if current_value == 0 || current_value == 1{
                found_addresses.push(address);
                // }
            }

            found_addresses
        }

        fn search_all_specific_values(core: &mut CoreMutRef) -> Vec<u32> {
            let segment = -1; // Default segment, adjust as necessary
            let mut found_addresses = Vec::new();

            // Define the memory range to search - adjust based on your game's addressable space
            let start_address = 0x02000000; // Start of EWRAM, commonly used in GBA games
            let end_address = 0x02040000; // End of EWRAM

            // Search through the address range for the value 196
            for address in (start_address..end_address).step_by(2) {
                // Changed step_by from 4 to 2
                let current_value = core.raw_read_16(address, segment);
                if current_value == 196 {
                    found_addresses.push(address);
                }
            }

            found_addresses
        }

        // Add these as state variables or within a struct if needed.

        // fn display_health_state(core: &mut CoreMutRef, is_offerer: bool) {
        //     // Define addresses for potential health values
        //     let server_health_address = 0x0203A9D4; // Server side health
        //     let client_health_address = 0x0203AAAC; // Client side health
        //     let segment = -1; // Default segment; adjust if necessary

        //     // Determine labels based on whether the instance is the server or the client
        //     let (player_label, opponent_label, player_health_address, opponent_health_address) = if is_offerer {
        //         // Server: Player is at 0x0203A9D4, Opponent is at 0x0203AAAC
        //         (
        //             "Player Health",
        //             "Opponent Health",
        //             server_health_address,
        //             client_health_address,
        //         )
        //     } else {
        //         // Client: Opponent is at 0x0203A9D4, Player is at 0x0203AAAC
        //         (
        //             "Opponent Health",
        //             "Player Health",
        //             client_health_address,
        //             server_health_address,
        //         )
        //     };

        //     // Read current health values
        //     let current_player_health = core.raw_read_16(player_health_address, segment);
        //     let current_opponent_health = core.raw_read_16(opponent_health_address, segment);

        //     // Safety: Ensure safe access to the static variables
        //     unsafe {
        //         // Check if player's health has decreased (punishment)
        //         if current_player_health < LAST_PLAYER_HEALTH {
        //             let damage = LAST_PLAYER_HEALTH - current_player_health;
        //             // Record punishment in global PUNISHMENTS
        //             add_punishment(RewardPunishment { damage });
        //         }

        //         // Check if opponent's health has decreased (reward)
        //         if current_opponent_health < LAST_OPPONENT_HEALTH {
        //             let damage = LAST_OPPONENT_HEALTH - current_opponent_health;
        //             // Record reward in global REWARDS
        //             add_reward(RewardPunishment { damage });
        //         }

        //         // Update last known health values
        //         LAST_PLAYER_HEALTH = current_player_health;
        //         LAST_OPPONENT_HEALTH = current_opponent_health;

        //         set_player_health(current_player_health);
        //         set_enemy_health(current_opponent_health);
        //         //display player and enemy health
        //         //display if is player
        //         // println!("{}: {}", player_label, current_player_health);
        //         // println!("{}: {}", opponent_label, current_opponent_health);

        //         if (current_player_health == 0 && !current_opponent_health == 0
        //             || !current_player_health == 0 && current_opponent_health == 0)
        //         {
        //             //close the program
        //             // Exit the application after sending the winner message
        //             std::process::exit(0);
        //         }
        //     }
        // }

        // Shared storage for initial boolean states
        static INITIAL_BOOLEAN_STATES: LazyLock<Mutex<HashMap<u32, u16>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

        // Initialize the log file with thread-safe access
        let log_file = Arc::new(Mutex::new(
            File::create("/home/lee/log.txt").expect("Failed to create log file"),
        ));
        // Clone the Arc to pass into the closure
        let log_file_clone = Arc::clone(&log_file);

        let window_state_addresses = vec![
            (0x02035288, 0, 255), //success! 0=closed, 255 = open
        ];

        // Clone for use inside the callback
        let window_state_addresses = Arc::new(window_state_addresses);

        // Shared storage for boolean addresses
        let boolean_addresses = Arc::new(Mutex::new(Vec::new()));
        let boolean_addresses_clone = Arc::clone(&boolean_addresses);
        // Initialize toggle state variables for Command 4
        let logging_enabled = Arc::new(AtomicBool::new(false));
        let command4_prev_state = Arc::new(AtomicBool::new(false));

        // // Clone these for use inside the closure
        let logging_enabled_clone = Arc::clone(&logging_enabled);
        let command4_prev_state_clone = Arc::clone(&command4_prev_state);

        let commands_enabled = Arc::new(AtomicBool::new(true));
        let local_player_index_for_frame = Arc::clone(&local_player_index_arc);
        let mut is_left_player = -1;



        // Wrap the health variables in Arc<Mutex<T>>
        let last_player_health = Arc::new(Mutex::new(0u16));
        let last_opponent_health = Arc::new(Mutex::new(0u16));

        // Clone the Arc instances to pass into the closure
        let last_player_health_clone = Arc::clone(&last_player_health);
        let last_opponent_health_clone = Arc::clone(&last_opponent_health);

        // Set the frame callback
        thread.set_frame_callback({
            let completion_token = completion_token.clone();
            let joyflags = joyflags.clone();
            let vbuf = vbuf.clone();
            let emu_tps_counter = emu_tps_counter.clone();
            let log_file = log_file_clone.clone(); // Clone the Arc for use inside the closure
                                                   // let boolean_addresses = boolean_addresses_clone.clone();
            let last_player_health = last_player_health_clone.clone();
            let last_opponent_health = last_opponent_health_clone.clone();

            move |mut core, video_buffer, mut thread_handle| {
                let mut is_left_player = is_left_player; // Create a mutable copy of is_left_player

                // Lock the Mutex to access and modify the health values
                let mut last_player_health = last_player_health.lock();
                let mut last_opponent_health = last_opponent_health.lock();

                let mut vbuf = vbuf.lock();
                vbuf.copy_from_slice(video_buffer);
                video::fix_vbuf_alpha(&mut vbuf);
                //joyflags.load(std::sync::atomic::Ordering::Relaxed)
                let joyflags_val = joyflags.load(std::sync::atomic::Ordering::Relaxed);
                core.set_keys(joyflags_val);
                set_local_input(joyflags_val as u16);
                emu_tps_counter.lock().mark();
                //log charge state 0x020369BD
                //enemy charge state 02036A10
                //enemy charge state 02036A74
                //enemy charge state 02036A78
                //enemy charge state 02036A80

                //player charge state 02036948
                //player charge state 020369AC
                //player charge state 020369B0
                //player charge state 020369BB
                // Access the local_player_index
                let local_player_index = local_player_index_for_frame.load(Ordering::SeqCst);

                //let is_offerer = local_player_index == 1;

                // if is_offerer {
                // } else {
                //     // println!("First Chip P2: {}", core.raw_read_16(0x203A9DA, -1));
                //     set_enemy_selected_chip(core.raw_read_16(0x203A9DA, -1));
                //     // println!("First Chip P1: {}", core.raw_read_16(0x203AAB2, -1));
                //     set_player_selected_chip(core.raw_read_16(0x203AAB2, -1));

                //     set_enemy_charge(core.raw_read_8(0x0203409D, -1) as u16);
                //     set_player_charge(core.raw_read_8(0x0203419D, -1) as u16);
                // }

                set_cust_gage(core.raw_read_8(0x20352A1, -1) as u16);

                set_player_emotion_state(core.raw_read_8(0x020352CC, -1) as u16);
                // //search all raw 16 values for ones that equal 196
                // let found_addresses = search_all_specific_values(&mut core);
                // println!("Found {} addresses with value 196", found_addresses.len());
                // //print out the addresses
                // for address in found_addresses.iter() {
                //     println!("Address: 0x{:08X}", address);
                // }

                // println!("Enemy Charge state: {}", core.raw_read_8(0x02036A10, -1));
                // println!("Player Charge state: {}", core.raw_read_8(0x0203419D, -1));
                //set enemy and player charge globally

                // println!("Player emotion state: {}", core.raw_read_8(0x020352CC, -1) as u16);

                // 0203A9E6 = 100 -> 20
                // 0203AA4C = 20 -> 140

                // 0203AB24 = 20 -> 140 (for other player)

                // set_player_game_emotion_state(core.raw_read_8(0x02035290, -1) as u16);

                // 2034136 x
                // 2034137 y
                // 2034236 x2
                // 2034236 y2
                //position of player 1
                let player_grid_positionX1 = core.raw_read_8(0x0203AA9C, -1);
                let player_grid_positionY1 = core.raw_read_8(0x0203AA9D, -1);
                // let player_grid_positionX1 = core.raw_read_8(0x02034136, -1);
                // let player_grid_positionY1 = core.raw_read_8(0x02034137, -1);
                // println!("Player grid position1: {}, {}", player_grid_positionX1, player_grid_positionY1);

                //position of player 2
                let player_grid_positionX2 = core.raw_read_8(0x020384FA, -1);
                let player_grid_positionY2 = core.raw_read_8(0x020384FB, -1);
                // let player_grid_positionX2 = core.raw_read_8(0x02034236, -1);
                // let player_grid_positionY2 = core.raw_read_8(0x02034237, -1);
                // println!("Player grid position2: {}, {}", player_grid_positionX2, player_grid_positionY2);

                set_player_grid_position((player_grid_positionX2 as u16, player_grid_positionY2 as u16));
                set_enemy_grid_position((player_grid_positionX1 as u16, player_grid_positionY1 as u16));

                // println!("Player game emotion state: {}", core.raw_read_8(0x02035290, -1) as u16);
                // println!("P2 emotion state: {}", core.raw_read_8(0x0203CE2C, -1) as u16);
                //set enemy emotion state

                set_selected_menu_index(core.raw_read_8(0x020364C7, -1) as u16);
                // println!("Selected menu index: {}", core.raw_read_8(0x020364C7, -1) as u16);

                set_selected_cross_index(core.raw_read_8(0x020364DB, -1) as u16);
                // println!("Selected cross index: {}", core.raw_read_8(0x020364DB, -1) as u16);

                //assign to 3x6 grid
                let grid_addresses = [
                    0x02039C06, 0x02039C26, 0x02039C46, 0x02039C66, 0x02039C86, 0x02039CA6, 
                    0x02039D06, 0x02039D26, 0x02039D46, 0x02039D66, 0x02039D86, 0x02039DA6, 
                    0x02039E06, 0x02039E26, 0x02039E46, 0x02039E66, 0x02039E86, 0x02039EA6,
                ];
                let grid_owner_addresses = [
                    0x02039C07, 0x02039C27, 0x02039C47, 0x02039C67, 0x02039C87, 0x02039CA7, 
                    0x02039D07, 0x02039D27, 0x02039D47, 0x02039D67, 0x02039D87, 0x02039DA7, 
                    0x02039E07, 0x02039E27, 0x02039E47, 0x02039E67, 0x02039E87, 0x02039EA7,
                ];
                let grid_occupy_addresses = [
                    0x02039C16, 0x02039C36, 0x02039C56, 0x02039C76, 0x02039C97, 0x02039CB7, 
                    0x02039D16, 0x02039D36, 0x02039D56, 0x02039D76, 0x02039D97, 0x02039DB7, 
                    0x02039E16, 0x02039E36, 0x02039E56, 0x02039E76, 0x02039E97, 0x02039EB7,
                ];

                //display values of grid
                let mut i = 0;
                for address in &grid_addresses {
                    let current_value = core.raw_read_8(*address, -1);
                    set_grid_state(i, current_value as u16);
                    i += 1;
                    // println!("0x{:08X},{}", address, current_value);
                }

                //display values of grid
                i = 0;
                for address in &grid_owner_addresses {
                    let current_value = core.raw_read_8(*address, -1);

                    set_grid_owner_state(i, current_value as u16);
                    i += 1;
                    // println!("0x{:08X},{}", address, current_value);
                }

                //display values of grid
                i = 0;
                for address in &grid_occupy_addresses {
                    let current_value = core.raw_read_8(*address, -1);
                    set_grid_occupy_state(i, current_value as u16);
                    i += 1;
                    // println!("0x{:08X},{}", address, current_value);
                }

                for i in 0..16 {
                    let address = 0x0203CDB0 + i as u32;
                    if i % 2 == 0 {
                        match set_chip_slot(i / 2, core.raw_read_8(address as u32, -1) as u16) {
                            Ok(_) => {
                                // println!("Chip slot {}: {}", i / 2, core.raw_read_8(address as u32, -1) as u16);
                            }
                            Err(e) => eprintln!("Failed to set chip slot {}: {:?}", i / 2, e),
                        }
                    } else {
                        match set_chip_code(i / 2, core.raw_read_8(address as u32, -1) as u16) {
                            Ok(_) => {
                                // println!("Chip code {}: {}", i / 2, core.raw_read_8(address as u32, -1) as u16);
                            }
                            Err(e) => eprintln!("Failed to set chip code {}: {:?}", i / 2, e),
                        }
                    }
                }

                set_chip_selected_count(core.raw_read_8(0x020364C8, -1) as u16);
                // println!("Chip selected count: {}", core.raw_read_8(0x020364C8, -1) as u16);
                //from 036508 through 03650C set the chip index selected
                for i in 0..5 {
                    let address = 0x02036508 + i as u32;
                    match set_selected_chip_index(i, core.raw_read_8(address as u32, -1) as u16) {
                        Ok(_) => {
                            // println!("Selected chip index {}: {}", i, core.raw_read_8(address as u32, -1) as u16);
                        }
                        Err(e) => eprintln!("Failed to set selected chip index {}: {:?}", i, e),
                    }
                }

                set_player_invis(core.raw_read_16(0x02038514, -1) as u16);
                // println!("Player invis: {}", core.raw_read_16(0x02038514, -1) as u16);


                set_player_trap((core.raw_read_8(0x2036720, -1)) as u16);
                // println!("Player trap: {}", core.raw_read_8(0x2036720, -1) as u16);


                //enemy trap? 020368A0 0203A9DA? 0203AAB8 02036844 0x02036730!   02034134!
                //enemy aura? 02046FF7

                set_enemy_trap(core.raw_read_8(0x02036730, -1) as u16);

                set_player_barrier(core.raw_read_8(0x2038506, -1) as u16);
                // println!("Player barrier: {}", core.raw_read_8(0x2038506, -1) as u16);

                set_player_aura(core.raw_read_8(0x20340E3, -1) as u16);
                // println!("Player aura: {}", core.raw_read_8(0x20340E3, -1) as u16);

                set_time_frozen( core.raw_read_8(0x02009878, -1) as u16);
                // println!("Time frozen: {}", core.raw_read_8(0x02009878, -1) as u16);



                set_beast_out_selectable(core.raw_read_8(0x0203664B, -1) as u16);
                // println!("Beast out selectable: {}", core.raw_read_8(0x0203664B, -1) as u16);

                set_chip_count_visible(core.raw_read_8(0x020047D6, -1) as u16);
                // println!("Chip count visible: {}", core.raw_read_8(0x020047D6, -1) as u16);

                set_inside_cross_window(core.raw_read_8(0x020364C2, -1) as u16);

                if completion_token.is_complete() {
                    thread_handle.pause();
                    //close application
                    std::process::exit(0);
                } else {
                    // display_health_state(&mut core, is_offerer);
                    let core_ref = &mut core;

                    // **Determine and Log the Assumed Window State for Each Address**
                    for &(address, expected_closed, expected_open) in window_state_addresses.iter() {
                        let current_value = core_ref.raw_read_8(address, -1);
                        let assumed_state = if current_value == expected_open {
                            "Open"
                        } else if current_value == expected_closed {
                            "Closed"
                        } else {
                            "Unknown"
                        };
                        set_is_player_inside_window(assumed_state == "Open");
                    }

                    //display 0x0203A9C4's value
                    //opponent's y position 261 517 773
                    // let current_value = core_ref.raw_read_16(0x0203A9C4, -1);

                    // println!("0x0203A9C4's value: {}", current_value);

                    // //display x position 220 panel 6, 180 panel 5, 140 panel 4, 100 panel 3, 60 panel 2, 20 panel 1
                    // let current_value = core_ref.raw_read_16(0x0203AA4C, -1);
                    // println!("0x0203AA4C's value: {}", current_value);

                    //0x02036A8C local player x
                    //0x0203AA9C local player y

                    let player_x = core_ref.raw_read_16(0x0203AA4C, -1);
                    let player_y = core_ref.raw_read_16(0x0203A9C4, -1);
                    let enemy_x = core_ref.raw_read_16(0x0203AB24, -1);
                    let enemy_y = core_ref.raw_read_16(0x0203AA9C, -1);

                    if is_left_player == -1 && player_x != 0 && player_y != 0 && enemy_x != 0 && enemy_y != 0 {
                        if player_x < enemy_x {
                            is_left_player = 1;
                        } else {
                            is_left_player = 0;
                        }
                    }
                    // println!("Enemy x: {}, Enemy y: {}", enemy_x, enemy_y);
                    // println!("Player x: {}, Player y: {}", player_x, player_y);

                    let server_health_address = 0x0203A9D4; // Server side health
                    let client_health_address = 0x0203AAAC; // Client side health

                    if local_player_index == 1 {
                        set_player_game_emotion_state(core_ref.raw_read_8(0x0203CE90, -1) as u16);
                        set_enemy_game_emotion_state(core_ref.raw_read_8(0x0203CE2C, -1) as u16);
                    } else {
                        set_player_game_emotion_state(core_ref.raw_read_8(0x0203CE2C, -1) as u16);
                        set_enemy_game_emotion_state(core_ref.raw_read_8(0x0203CE90, -1) as u16);
                    }

                    if is_left_player == 1 {

                        set_is_offerer(1);
                        set_enemy_position((enemy_x, enemy_y));
                        set_player_position((player_x, player_y));

                        // println!("First Chip P1: {}", core.raw_read_16(0x203A9DA, -1));
                        set_player_selected_chip(core_ref.raw_read_16(0x203A9DA, -1));
                        // println!("First Chip P2: {}", core.raw_read_16(0x203AAB2, -1));
                        set_enemy_selected_chip(core_ref.raw_read_16(0x203AAB2, -1));

                        set_player_charge(core_ref.raw_read_8(0x0203409D, -1) as u16);
                        set_enemy_charge(core_ref.raw_read_8(0x0203419D, -1) as u16);

                        let current_player_health = core_ref.raw_read_16(server_health_address, -1);
                        let current_opponent_health = core_ref.raw_read_16(client_health_address, -1);

                        // Check if player's health has decreased (punishment)
                        if current_player_health != *last_player_health && *last_player_health != 0 {
                            let damage = *last_player_health - current_player_health;
                            // Record punishment in global PUNISHMENTS
                            add_punishment(RewardPunishment { damage });
                        }

                        // Check if opponent's health has decreased (reward)
                        if current_opponent_health != *last_opponent_health && *last_opponent_health != 0 {
                            let damage = *last_opponent_health - current_opponent_health;
                            // Record reward in global REWARDS
                            add_reward(RewardPunishment { damage });
                        }

                        // Update last known health values
                        *last_player_health = current_player_health;
                        *last_opponent_health = current_opponent_health;

                        set_player_health(current_player_health);
                        set_enemy_health(current_opponent_health);

                        if (current_player_health == 0 && current_opponent_health != 0)
                            || (current_player_health != 0 && current_opponent_health == 0)
                        {
                            // Exit the application after sending the winner message
                            std::process::exit(0);
                        }
                    } else {
                        set_is_offerer(0);
                        set_enemy_position((player_x, player_y));
                        set_player_position((enemy_x, enemy_y));

                        // println!("First Chip P1: {}", core.raw_read_16(0x203A9DA, -1));
                        set_enemy_selected_chip(core_ref.raw_read_16(0x203A9DA, -1));
                        // println!("First Chip P2: {}", core.raw_read_16(0x203AAB2, -1));
                        set_player_selected_chip(core_ref.raw_read_16(0x203AAB2, -1));

                        set_enemy_charge(core_ref.raw_read_8(0x0203409D, -1) as u16);
                        set_player_charge(core_ref.raw_read_8(0x0203419D, -1) as u16);

                        let current_player_health = core_ref.raw_read_16(client_health_address, -1);
                        let current_opponent_health = core_ref.raw_read_16(server_health_address, -1);

                        // Check if player's health has decreased (punishment)
                        if current_player_health != *last_player_health && *last_player_health != 0 {
                            let damage = *last_player_health - current_player_health;
                            // Record punishment in global PUNISHMENTS
                            add_punishment(RewardPunishment { damage });
                        }

                        // Check if opponent's health has decreased (reward)
                        if current_opponent_health != *last_opponent_health && *last_opponent_health != 0 {
                            let damage = *last_opponent_health - current_opponent_health;
                            // Record reward in global REWARDS
                            add_reward(RewardPunishment { damage });
                        }
                        // Update last known health values
                        *last_player_health = current_player_health;
                        *last_opponent_health = current_opponent_health;

                        set_player_health(current_player_health);
                        set_enemy_health(current_opponent_health);

                        if (current_player_health == 0 && current_opponent_health != 0)
                            || (current_player_health != 0 && current_opponent_health == 0)
                        {
                            // Exit the application after sending the winner message
                            std::process::exit(0);
                        }
                    }

                    if commands_enabled.load(std::sync::atomic::Ordering::Relaxed) {
                        let joyflags_val = joyflags.load(std::sync::atomic::Ordering::Relaxed);
                        // Command 1 (Set all values, triggered by 0x00000200)
                        if joyflags_val & 0x00000200 != 0 {
                            // println!("Set Command - Populating boolean addresses");
                            let mut addresses = boolean_addresses.lock();
                            let mut initial_states = INITIAL_BOOLEAN_STATES.lock();

                            // Clear previous boolean addresses and states
                            addresses.clear();
                            initial_states.clear();

                            // Search for boolean values in memory and populate boolean_addresses
                            let found_addresses = search_all_boolean_values(core_ref);
                            addresses.extend(found_addresses);

                            // Store the initial values of those addresses in INITIAL_BOOLEAN_STATES
                            for &address in addresses.iter() {
                                //only get values that are 0

                                let current_value = core_ref.raw_read_16(address, -1);
                                // if current_value == 0 {
                                initial_states.insert(address, current_value);
                                // }
                                println!("Captured Address: 0x{:08X}, Initial Value: {}", address, current_value);
                            }

                            // println!("Set Command - Populated {} boolean addresses.", addresses.len());
                        }

                        // Command 2 (Prune the list, triggered by 0x00000100)
                        if joyflags_val & 0x00000100 != 0 {
                            // println!("Prune Command - Pruning boolean addresses");
                            let mut addresses = boolean_addresses.lock();
                            let mut initial_states = INITIAL_BOOLEAN_STATES.lock();
                            addresses.retain(|&address| {
                                let current_value = core_ref.raw_read_16(address, -1);
                                if let Some(&initial_value) = initial_states.get(&address) {
                                    if current_value != initial_value {
                                        // Value has changed, prune it from the list
                                        println!(
                                            "Pruned Address: 0x{:08X}, Initial Value: {}, Current Value: {}",
                                            address, initial_value, current_value
                                        );
                                        return false;
                                    }
                                }
                                true
                            });

                            // println!("Prune Command - Remaining {} boolean addresses.", addresses.len());
                        }

                        // Command 2 (Prune the list, triggered by 0x00000100)
                        if joyflags_val & 0x00000002 != 0 {
                            // println!("Prune Command - Pruning same boolean addresses");
                            let mut addresses = boolean_addresses.lock();
                            let mut initial_states = INITIAL_BOOLEAN_STATES.lock();
                            addresses.retain(|&address| {
                                let current_value = core_ref.raw_read_16(address, -1);
                                if let Some(&initial_value) = initial_states.get(&address) {
                                    if current_value == initial_value {
                                        // Value has changed, prune it from the list
                                        println!(
                                            "Pruned Address: 0x{:08X}, Initial Value: {}, Current Value: {}",
                                            address, initial_value, current_value
                                        );
                                        return false;
                                    }
                                }
                                true
                            });

                            // println!("Prune Command - Remaining {} boolean addresses.", addresses.len());
                        }

                        // Command 3 (Compare the current values, triggered by 0x00000008)
                        if joyflags_val & 0x00000008 != 0 {
                            // println!("Compare Command - Checking different current values");
                            let addresses = boolean_addresses.lock();
                            let mut initial_states = INITIAL_BOOLEAN_STATES.lock();

                            for &address in addresses.iter() {
                                let current_value = core_ref.raw_read_16(address, -1);
                                //update initial value to the current value
                                if let Some(initial_value) = initial_states.get_mut(&address) {
                                    *initial_value = current_value;
                                }
                                // if let Some(&initial_value) = initial_states.get(&address) {
                                //     // Only display if the current value has changed compared to the initial value
                                //     if current_value != initial_value {
                                //         println!(
                                //             "Compare Command - Address: 0x{:08X}, Initial Value: {}, Current Value: {}",
                                //             address, initial_value, current_value
                                //         );
                                //     }
                                // }
                            }
                        }
                        // Command 3 (Compare the current values, triggered by 0x00000008)
                        // if joyflags_val & 0x00000001 != 0 {
                        //    // Lock the boolean_addresses and log_file for safe access
                        //     let addresses = boolean_addresses.lock();
                        //     let mut file = log_file.lock();

                        //     // Iterate over each address and write to the file
                        //     for &address in addresses.iter() {
                        //         if let Err(e) = writeln!(file, "0x{:08X}", address) {
                        //             eprintln!("Failed to write address 0x{:08X} to log file: {}", address, e);
                        //         }
                        //     }

                        //     // Optionally, flush the file to ensure data is written immediately
                        //     if let Err(e) = file.flush() {
                        //         eprintln!("Failed to flush log file: {}", e);
                        //     }

                        // }
                        // Handle Command 4 as a toggle
                        {
                            // Check the current state of Command 4 bit (0x00000004)
                            let command4_current = (joyflags_val & 0x00000004) != 0;
                            let command4_prev = command4_prev_state.load(Ordering::Relaxed);

                            // Detect rising edge: Command 4 bit was not set in the previous frame but is set now
                            if command4_current && !command4_prev {
                                // Toggle the logging_enabled state
                                let new_state = !logging_enabled.load(Ordering::Relaxed);
                                logging_enabled.store(new_state, Ordering::Relaxed);
                                if new_state {
                                    println!("Command 4 Toggle: Logging ENABLED.");
                                } else {
                                    println!("Command 4 Toggle: Logging DISABLED.");
                                }
                            }

                            // Update the previous state for the next frame
                            command4_prev_state.store(command4_current, Ordering::Relaxed);
                        }

                        // If logging is enabled, perform the logging
                        if logging_enabled.load(Ordering::Relaxed) {
                            println!("Command 4 Action: Logging Remaining Boolean Addresses and Their Current Values.");
                            let addresses = boolean_addresses.lock();
                            let initial_states = INITIAL_BOOLEAN_STATES.lock();

                            for &address in addresses.iter() {
                                let current_value = core_ref.raw_read_8(address, -1);
                                if let Some(&initial_value) = initial_states.get(&address) {
                                    println!(
                                        "Address: 0x{:08X}, Initial Value: {}, Current Value: {}",
                                        address, initial_value, current_value
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });

        let own_setup = {
            let assets = local_game.load_rom_assets(local_rom, &local_save.as_raw_wram(), local_patch_overrides)?;
            Some(Setup {
                game_lang: local_patch_overrides
                    .language
                    .clone()
                    .unwrap_or_else(|| crate::game::region_to_language(local_game.gamedb_entry().region)),
                save: local_save,
                assets,
            })
        };
        let mut opponent_setup = if true {
            let assets = remote_game.load_rom_assets(remote_rom, &remote_save.as_raw_wram(), remote_patch_overrides)?;
            Some(Setup {
                game_lang: remote_patch_overrides
                    .language
                    .clone()
                    .unwrap_or_else(|| crate::game::region_to_language(remote_game.gamedb_entry().region)),
                save: remote_save,
                assets,
            })
        } else {
            None
        };

        // After initializing `own_setup` and `opponent_setup`
        let own_navi_cust_ids = if let Some(ref setup) = own_setup {
            extract_navi_cust_ids(&*setup.save, setup.assets.as_ref())
        } else {
            Vec::new()
        };

        let opponent_navi_cust_ids = if let Some(ref setup) = opponent_setup {
            extract_navi_cust_ids(&*setup.save, setup.assets.as_ref())
        } else {
            Vec::new()
        };

        // Log the Navi Cust IDs
        println!("Local Player Navi Cust IDs: {:?}", own_navi_cust_ids);
        println!("Opponent Player Navi Cust IDs: {:?}", opponent_navi_cust_ids);

        set_player_navi_cust_parts(own_navi_cust_ids);
        set_enemy_navi_cust_parts(opponent_navi_cust_ids);

        // **Extract and Print Chips**
        let own_chips = if let Some(ref setup) = own_setup {
            extract_chips(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        let opponent_chips = if let Some(ref setup) = opponent_setup {
            extract_chips(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        // **Extract and Print Chips**
        let own_tag_chips = if let Some(ref setup) = own_setup {
            extract_tag_indices(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        let opponent_tag_chips = if let Some(ref setup) = opponent_setup {
            extract_tag_indices(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        // Extract regular chip index
        let own_reg_chip_index = if let Some(ref setup) = own_setup {
            extract_reg_index(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(60)
        };

        let opponent_reg_chip_index = if let Some(ref setup) = opponent_setup {
            extract_reg_index(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(60)
        };

        let own_reg_chip_index = own_reg_chip_index.unwrap_or(60);
        set_player_reg_chip(own_reg_chip_index as u16);
        let opponent_reg_chip_index = opponent_reg_chip_index.unwrap_or(60);
        set_enemy_reg_chip(opponent_reg_chip_index as u16);

        // Log the Chips
        println!("Local Player Chips:");
        if let Ok(chips) = own_chips {
            let mut index = 0;
            for chip in chips {
                // println!("  ID: {}, Code: {}", chip.id, chip.code);
                if let Err(e) = set_player_chip_folder(index, chip.id as u16) {
                    eprintln!("Failed to set player chip folder: {:?}", e);
                }
                match set_player_code_folder(index, chip.code as u16) {
                    Ok(_) => {}
                    Err(e) => eprintln!("Failed to set player code folder: {:?}", e),
                }
                index += 1;
            }
        }

        println!("Opponent Player Chips:");
        if let Ok(chips) = opponent_chips {
            let mut index = 0;
            for chip in chips {
                if let Err(e) = set_enemy_chip_folder(index, chip.id as u16) {
                    eprintln!("Failed to set enemy chip folder: {:?}", e);
                }
                if let Err(e) = set_enemy_code_folder(index, chip.code as u16) {
                    eprintln!("Failed to set enemy code folder: {:?}", e);
                }
                // println!("  ID: {}, Code: {}", chip.id, chip.code);
                index += 1;
            }
        }

        // Log the Tag Chips
        println!("Local Player Tag Chips:");
        if let Ok(tag_chips) = own_tag_chips {
            let mut index = 0;
            for tag_chip in tag_chips {
                // println!("Tag Chip: {}", tag_chip);
                set_player_tag_folder(index, tag_chip as u16);
                index += 1;
            }
        }
        println!("Opponent Player Tag Chips:");
        if let Ok(tag_chips) = opponent_tag_chips {
            let mut index = 0;
            for tag_chip in tag_chips {
                // println!("Tag Chip: {}", tag_chip);
                set_enemy_tag_folder(index, tag_chip as u16);
                index += 1;
            }
        }
        opponent_setup = None;

        Ok(Session {
            start_time: std::time::SystemTime::now(),
            game_info: GameInfo {
                game: local_game,
                patch: local_patch,
            },
            vbuf,
            _audio_binding: audio_binding,
            thread,
            joyflags,
            mode: Mode::PvP(PvP {
                match_,
                cancellation_token,
                _peer_conn: peer_conn,
                latency_counter,
            }),
            completion_token,
            pause_on_next_frame: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            own_setup,
            opponent_setup,
        })
    }

    pub fn new_singleplayer(
        audio_binder: audio::LateBinder,
        game: &'static (dyn game::Game + Send + Sync),
        patch: Option<(String, semver::Version)>,
        rom: &[u8],
        save_file: std::fs::File,
        emu_tps_counter: Arc<Mutex<stats::Counter>>,
    ) -> Result<Self, anyhow::Error> {
        let mut core = mgba::core::Core::new_gba("tango")?;
        core.enable_video_buffer();

        core.as_mut().load_rom(mgba::vfile::VFile::from_vec(rom.to_vec()))?;

        let save_vf = mgba::vfile::VFile::from_file(save_file);

        core.as_mut().load_save(save_vf)?;

        let joyflags = Arc::new(std::sync::atomic::AtomicU32::new(0));

        let hooks = tango_pvp::hooks::hooks_for_gamedb_entry(game.gamedb_entry()).unwrap();
        hooks.patch(core.as_mut());

        let thread = mgba::thread::Thread::new(core);

        thread.start()?;
        thread.handle().lock_audio().sync_mut().set_fps_target(EXPECTED_FPS);

        let audio_binding = audio_binder.bind(Some(Box::new(audio::MGBAStream::new(
            thread.handle(),
            audio_binder.sample_rate(),
        ))))?;

        let pause_on_next_frame = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let vbuf = Arc::new(Mutex::new(vec![
            0u8;
            (mgba::gba::SCREEN_WIDTH * mgba::gba::SCREEN_HEIGHT * 4)
                as usize
        ]));
        thread.set_frame_callback({
            let joyflags = joyflags.clone();
            let vbuf = vbuf.clone();
            let emu_tps_counter = emu_tps_counter.clone();
            let pause_on_next_frame = pause_on_next_frame.clone();
            move |mut core, video_buffer, mut thread_handle| {
                let mut vbuf = vbuf.lock();
                vbuf.copy_from_slice(video_buffer);
                video::fix_vbuf_alpha(&mut vbuf);
                core.set_keys(joyflags.load(std::sync::atomic::Ordering::Relaxed));
                emu_tps_counter.lock().mark();

                if pause_on_next_frame.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    thread_handle.pause();
                }
            }
        });
        Ok(Session {
            start_time: std::time::SystemTime::now(),
            game_info: GameInfo { game, patch },
            vbuf,
            _audio_binding: audio_binding,
            thread,
            joyflags,
            mode: Mode::SinglePlayer(SinglePlayer {}),
            pause_on_next_frame,
            completion_token: tango_pvp::hooks::CompletionToken::new(),
            own_setup: None,
            opponent_setup: None,
        })
    }

    pub fn new_replayer(
        audio_binder: audio::LateBinder,
        game: &'static (dyn game::Game + Send + Sync),
        patch: Option<(String, semver::Version)>,
        rom: &[u8],
        emu_tps_counter: Arc<Mutex<stats::Counter>>,
        replay: &tango_pvp::replay::Replay,
        roms: HashMap<(String, u8), Vec<u8>>, // Update the type here
                                              // patches: HashMap<String, Patch>,
                                              // patches_path: std::path::PathBuf,
    ) -> Result<Self, anyhow::Error> {
        let mut core = mgba::core::Core::new_gba("tango")?;
        core.enable_video_buffer();

        core.as_mut().load_rom(mgba::vfile::VFile::from_vec(rom.to_vec()))?;

        let hooks = tango_pvp::hooks::hooks_for_gamedb_entry(game.gamedb_entry()).unwrap();
        hooks.patch(core.as_mut());

        let completion_token = tango_pvp::hooks::CompletionToken::new();

        let replay_is_complete = replay.is_complete;
        let input_pairs = replay.input_pairs.clone();
        let stepper_state = tango_pvp::stepper::State::new(
            (replay.metadata.match_type as u8, replay.metadata.match_subtype as u8),
            replay.local_player_index,
            input_pairs.clone(),
            0,
            Box::new({
                let completion_token = completion_token.clone();
                move || {
                    completion_token.complete();
                }
            }),
        );
        let mut traps = hooks.common_traps();

        traps.extend(
            hooks
                .stepper_traps(stepper_state.clone())
                .into_iter()
                .map(|(addr, original_func)| {
                    let stepper_state_clone = stepper_state.clone();
                    (
                        addr,
                        Box::new(move |core: mgba::core::CoreMutRef<'_>| {
                            original_func(core);

                            // Capture local input
                            let stepper_inner = stepper_state_clone.lock_inner();
                            if let Some(input_pair) = stepper_inner.peek_input_pair() {
                                let local_input = input_pair.local.joyflags;

                                // Add the local input to the global list
                                set_local_input(local_input);

                                // Print or log if necessary for debugging
                                // println!(
                                //     "At address {:08X}, Local Input: {:016b}",
                                //     addr, local_input
                                // );
                            }
                        }) as Box<dyn Fn(mgba::core::CoreMutRef<'_>)>,
                    )
                }),
        );
        traps.extend(hooks.stepper_replay_traps());

        core.set_traps(traps);

        let thread = mgba::thread::Thread::new(core);

        thread.start()?;
        thread.handle().pause();
        thread.handle().lock_audio().sync_mut().set_fps_target(EXPECTED_FPS);

        let audio_binding = audio_binder.bind(Some(Box::new(audio::MGBAStream::new(
            thread.handle(),
            audio_binder.sample_rate(),
        ))))?;

        let local_state = replay.local_state.clone();
        let remote_state = replay.remote_state.clone();
        let local_player_index = replay.local_player_index;

        thread.handle().run_on_core(move |mut core| {
            core.load_state(&local_state).expect("load state");
        });
        thread.handle().unpause();

        let pause_on_next_frame = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let vbuf = Arc::new(Mutex::new(vec![
            0u8;
            (mgba::gba::SCREEN_WIDTH * mgba::gba::SCREEN_HEIGHT * 4)
                as usize
        ]));

        // Add these as state variables or within a struct if needed.
        static mut LAST_PLAYER_HEALTH: u16 = 0;
        static mut LAST_OPPONENT_HEALTH: u16 = 0;
        static mut HEALTH_INITIALIZED: bool = false; // Flag to check if health has been properly initialized
        static HEALTH_ADDRESSES: LazyLock<Mutex<HashMap<u32, u16>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

        static PLAYER_HEALTH_ADDRESS: Mutex<Option<u32>> = Mutex::new(None);
        static OPPONENT_HEALTH_ADDRESS: Mutex<Option<u32>> = Mutex::new(None);
        static ADDRESSES_SET: Mutex<bool> = Mutex::new(false);

        fn display_health_state(core: &mut CoreMutRef, player_address: u32, opponent_address: u32) {
            let segment = -1; // Default segment; adjust if necessary
                              // Read current health values
            let current_player_health = core.raw_read_16(player_address, segment);
            let current_opponent_health = core.raw_read_16(opponent_address, segment);

            // // Determine labels based on whether the instance is the server or the client
            // let (player_label, opponent_label, player_health_address, opponent_health_address) = if is_offerer {
            //     // Server: Player is at 0x0203A9D4, Opponent is at 0x0203AAAC
            //     ("Player Health", "Opponent Health", server_health_address, client_health_address)
            // } else {
            //     // Client: Opponent is at 0x0203A9D4, Player is at 0x0203AAAC
            //     ("Opponent Health", "Player Health", client_health_address, server_health_address)
            // };

            // Read current health values
            // let current_player_health = core.raw_read_16(player_health_address, segment);
            // let current_opponent_health = core.raw_read_16(opponent_health_address, segment);

            // Safety: Ensure safe access to the static variables
            unsafe {
                // Check if health values are greater than 0 for the first time
                if !HEALTH_INITIALIZED && current_player_health > 0 && current_opponent_health > 0 {
                    HEALTH_INITIALIZED = true; // Set the flag to true, start checking for game state changes
                }

                // If health values are initialized, start monitoring for winner or damage changes
                if HEALTH_INITIALIZED {
                    // Check if player's health has decreased (punishment)
                    if current_player_health < LAST_PLAYER_HEALTH && LAST_PLAYER_HEALTH != 0 {
                        let damage = LAST_PLAYER_HEALTH - current_player_health;
                        // Record punishment in global PUNISHMENTS
                        add_punishment(RewardPunishment { damage });
                    }

                    // Check if opponent's health has decreased (reward)
                    if current_opponent_health < LAST_OPPONENT_HEALTH && LAST_OPPONENT_HEALTH != 0 {
                        let damage = LAST_OPPONENT_HEALTH - current_opponent_health;
                        // Record reward in global REWARDS
                        add_reward(RewardPunishment { damage });
                    }

                    // Check for game end and set the winner
                    if current_player_health == 0 {
                        //log health and addresses
                        println!("Player Health: {}", current_player_health);
                        println!("Opponent Health: {}", current_opponent_health);
                        println!("Player Health Address: 0x{:08X}", player_address);
                        println!("Opponent Health Address: 0x{:08X}", opponent_address);
                        set_winner(false); // Player lost
                    } else if current_opponent_health == 0 {
                        //log health and addresses
                        println!("Player Health: {}", current_player_health);
                        println!("Opponent Health: {}", current_opponent_health);
                        println!("Player Health Address: 0x{:08X}", player_address);
                        println!("Opponent Health Address: 0x{:08X}", opponent_address);
                        set_winner(true); // Player won
                    }
                }
                // Update last known health values
                LAST_PLAYER_HEALTH = current_player_health;
                LAST_OPPONENT_HEALTH = current_opponent_health;
                set_player_health(current_player_health);
                set_enemy_health(current_opponent_health);
                // println!("Player Health: {}", current_player_health);
                // println!("Opponent Health: {}", current_opponent_health);
                set_player_charge(core.raw_read_8(0x02036948, -1) as u16);
                set_enemy_charge(core.raw_read_8(0x02036A10, -1) as u16);
            }
        }

        // Function to search for all potential health addresses
        fn search_all_health_values(core: &mut CoreMutRef, value_to_find: u8) -> Vec<u32> {
            let segment = -1; // Default segment, adjust as necessary
            let mut found_addresses = Vec::new();

            // Define the memory range to search - adjust based on your game's addressable space
            let start_address = 0x02000000; // Start of EWRAM, commonly used in GBA games
            let end_address = 0x02080000; // End of EWRAM

            // Search through the address range for the specified value
            for address in (start_address..end_address).step_by(4) {
                let current_value = core.raw_read_8(address, segment);
                if current_value == value_to_find {
                    found_addresses.push(address);
                }
            }

            found_addresses
        }

        // for setting the player index
        set_frame_count(0);

        thread.set_frame_callback({
            let vbuf = vbuf.clone();
            let emu_tps_counter = emu_tps_counter.clone();
            let completion_token = completion_token.clone();
            let stepper_state = stepper_state.clone();
            let pause_on_next_frame = pause_on_next_frame.clone();
            let local_player_index = replay.local_player_index;
            move |mut core, video_buffer, mut thread_handle| {
                let mut vbuf = vbuf.lock();
                vbuf.copy_from_slice(video_buffer);
                video::fix_vbuf_alpha(&mut vbuf);
                emu_tps_counter.lock().mark();

                //get frame count
                let currentFrame = get_frame_count();
                // println!("Replayer created, loading setup data");

                set_frame_count(currentFrame + 1);

                if !replay_is_complete && stepper_state.lock_inner().input_pairs_left() == 0 {
                    completion_token.complete();
                }

                if pause_on_next_frame.swap(false, std::sync::atomic::Ordering::SeqCst)
                    || completion_token.is_complete()
                {
                    thread_handle.pause();
                    // Force exit the application when the replay is complete
                    if completion_token.is_complete() {
                        println!("Replay completed. Exiting application...");
                        std::process::exit(0); // Exit the application with code 0 (normal exit)
                    }
                } else {
                    let core_ref = &mut core;

                    // Print local player index
                    // println!("Local Player Index: {}", local_player_index);
                    let is_offerer = local_player_index == 1;

                    // Check if the addresses have already been set
                    let mut addresses_set = ADDRESSES_SET.lock();
                    if !*addresses_set {
                        // Define addresses for potential health values
                        let player_health_address = 0x020F52A4; // Example address
                        let segment = -1; // Default segment; adjust if necessary
                        let possible_addresses = [0x0203AAAC, 0x02B7AAAC, 0x0257AAAC, 0x0203A9D4, 0x0293A9D4];

                        // //debug print all possible healths
                        // for &address in &possible_addresses {
                        //     let current_value = core_ref.raw_read_16(address, segment);
                        //     println!("Possible Health Address: 0x{:08X}, Value: {}", address, current_value);
                        // }

                        // Read player health directly
                        let player_health = core_ref.raw_read_16(player_health_address, segment);
                        if player_health == 0 || player_health > 4000 {
                            return;
                        }

                        // Log values for all possible opponent addresses
                        for &address in &possible_addresses {
                            let current_value = core_ref.raw_read_16(address, segment);

                            // If the value isn't equal to the player's health, it's considered the opponent's health
                            if current_value != player_health {
                                *OPPONENT_HEALTH_ADDRESS.lock() = Some(address);
                                break;
                            }
                        }

                        // Check if opponent health address was found
                        if let Some(opponent_health_address) = *OPPONENT_HEALTH_ADDRESS.lock() {
                            // Continue with the rest of your logic

                            // Get the opponent's current health
                            let opponent_health = core_ref.raw_read_16(opponent_health_address, segment);

                            // Find an address from the possible addresses that is NOT the opponent's health
                            for &address in &possible_addresses {
                                let current_value = core_ref.raw_read_16(address, segment);
                                if current_value != opponent_health {
                                    *PLAYER_HEALTH_ADDRESS.lock() = Some(address);
                                    //check if this address is index 0-2 in the possible addresses array
                                    set_player_health_index(0);
                                    for i in 0..2 {
                                        if possible_addresses[i] == address {
                                            set_player_health_index(1);
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }

                            // Only stop trying to set the values once we know the player and enemy health aren't the same
                            if player_health != opponent_health {
                                //set player health index if the health address that was set to player health was index 0-2
                                *addresses_set = true;
                            }
                        } else {
                            // println!("Could not determine opponent's health address.");
                            return;
                        }
                    }

                    let window_state_addresses = vec![
                        (0x02035288, 0, 255), //success! 0=closed, 255 = open
                    ];

                    // Retrieve the health addresses
                    let player_health_addr = *PLAYER_HEALTH_ADDRESS.lock();
                    let opponent_health_addr = *OPPONENT_HEALTH_ADDRESS.lock();
                    // set_player_charge(core_ref.raw_read_8(0x02036948, -1) as u16);
                    // set_enemy_charge(core_ref.raw_read_8(0x02036A10, -1) as u16);

                    set_cust_gage(core_ref.raw_read_8(0x20352A1, -1) as u16);
                    set_player_emotion_state(core_ref.raw_read_8(0x020352CC, -1) as u16);
                    // println!("Player window emotion state: {}", core_ref.raw_read_8(0x020352CC, -1) as u16);

                    set_player_game_emotion_state(core_ref.raw_read_8(0x02035290, -1) as u16);
                    // println!(
                    //     "Player game emotion state: {}",
                    //     core_ref.raw_read_8(0x02035290, -1) as u16
                    // );

                    set_enemy_game_emotion_state(core_ref.raw_read_8(0x02035291, -1) as u16);

                    //values for the grid state (3x6) 8bit
                    //0x02039C06	0x02039C26	0x02039C46	0x02039C66	0x02039C86	0x02039CA6
                    //0x02039D06	0x02039D26	0x02039D46	0x02039D66	0x02039D86	0x02039DA6
                    //0x02039E06	0x02039E26	0x02039E46	0x02039E66	0x02039E86	0x02039EA6

                    //assign to 3x6 grid
                    let grid_addresses = [
                        0x02039C06, 0x02039C26, 0x02039C46, 0x02039C66, 0x02039C86, 0x02039CA6, 0x02039D06, 0x02039D26,
                        0x02039D46, 0x02039D66, 0x02039D86, 0x02039DA6, 0x02039E06, 0x02039E26, 0x02039E46, 0x02039E66,
                        0x02039E86, 0x02039EA6,
                    ];
                    let grid_owner_addresses = [
                        0x02039C07, 0x02039C27, 0x02039C47, 0x02039C67, 0x02039C87, 0x02039CA7, 0x02039D07, 0x02039D27,
                        0x02039D47, 0x02039D67, 0x02039D87, 0x02039DA7, 0x02039E07, 0x02039E27, 0x02039E47, 0x02039E67,
                        0x02039E87, 0x02039EA7,
                    ];

                    //display values of grid
                    let mut i = 0;
                    for address in &grid_addresses {
                        let current_value = core_ref.raw_read_8(*address, -1);

                        set_grid_state(i, current_value as u16);
                        i += 1;
                        // println!("0x{:08X},{}", address, current_value);
                    }
                    i = 0;
                    //display values of grid
                    for address in &grid_owner_addresses {
                        let current_value = core_ref.raw_read_8(*address, -1);
                        set_grid_owner_state(i, current_value as u16);
                        i += 1;
                        // println!("0x{:08X},{}", address, current_value);
                    }

                    //search for values that are equal to 9
                    // let possible_addresses = search_all_health_values(core_ref, 19);
                    // //output addresses as hex
                    // for address in &possible_addresses {
                    //     println!("0x{:08X},", address);
                    // }
                    // let addresses = [

                    // 0x02010B1C,
                    // 0x0203EAF4,
                    // 0x02050B1C,
                    // 0x0207EAF4,

                    // ];

                    // //display values of addresses 8
                    // for address in &addresses {
                    //     let current_value = core_ref.raw_read_8(*address, -1);
                    //     println!("0x{:08X},{}", address, current_value);
                    // }

                    //set enemy emotion state

                    set_selected_menu_index(core_ref.raw_read_8(0x020364C7, -1) as u16);
                    // println!("Selected menu index: {}", core_ref.raw_read_8(0x020364C7, -1) as u16);

                    set_selected_cross_index(core_ref.raw_read_8(0x020364DB, -1) as u16);
                    // println!("Selected cross index: {}", core_ref.raw_read_8(0x020364DB, -1) as u16);

                    for i in 0..16 {
                        let address = 0x0203CDB0 + i as u32;
                        if i % 2 == 0 {
                            match set_chip_slot(i / 2, core_ref.raw_read_8(address as u32, -1) as u16) {
                                Ok(_) => {
                                    // println!("Chip slot {}: {}", i / 2, core_ref.raw_read_8(address as u32, -1) as u16);
                                }
                                Err(e) => eprintln!("Failed to set chip slot {}: {:?}", i / 2, e),
                            }
                        } else {
                            match set_chip_code(i / 2, core_ref.raw_read_8(address as u32, -1) as u16) {
                                Ok(_) => {
                                    // println!("Chip code {}: {}", i / 2, core_ref.raw_read_8(address as u32, -1) as u16);
                                }
                                Err(e) => eprintln!("Failed to set chip code {}: {:?}", i / 2, e),
                            }
                        }
                    }

                    set_chip_selected_count(core_ref.raw_read_8(0x020364C8, -1) as u16);
                    // println!("Chip selected count: {}", core_ref.raw_read_8(0x020364C8, -1) as u16);
                    //from 036508 through 03650C set the chip index selected
                    for i in 0..5 {
                        let address = 0x02036508 + i as u32;
                        match set_selected_chip_index(i, core_ref.raw_read_8(address as u32, -1) as u16) {
                            Ok(_) => {
                                // println!("Selected chip index {}: {}", i, core_ref.raw_read_8(address as u32, -1) as u16);
                            }
                            Err(e) => eprintln!("Failed to set selected chip index {}: {:?}", i, e),
                        }
                    }

                    set_beast_out_selectable(core_ref.raw_read_8(0x0203664B, -1) as u16);
                    // println!("Beast out selectable: {}", core_ref.raw_read_8(0x0203664B, -1) as u16);

                    set_chip_count_visible(core_ref.raw_read_8(0x020047D6, -1) as u16);
                    // println!("Chip count visible: {}", core_ref.raw_read_8(0x020047D6, -1) as u16);

                    set_inside_cross_window(core_ref.raw_read_8(0x020364C2, -1) as u16);

                    // Ensure the addresses are set before calling `display_health_state`
                    if let (Some(player_addr), Some(opponent_addr)) = (player_health_addr, opponent_health_addr) {
                        display_health_state(core_ref, player_addr, opponent_addr);

                        let core_ref = &mut core;

                        // **Determine and Log the Assumed Window State for Each Address**
                        for &(address, expected_closed, expected_open) in window_state_addresses.iter() {
                            let current_value = core_ref.raw_read_8(address, -1);
                            let assumed_state = if current_value == expected_open {
                                "Open"
                            } else if current_value == expected_closed {
                                "Closed"
                            } else {
                                "Unknown"
                            };
                            set_is_player_inside_window(assumed_state == "Open");
                        }

                        //display 0x0203A9C4's value
                        //opponent's y position 261 517 773
                        // let current_value = core_ref.raw_read_16(0x0203A9C4, -1);

                        // println!("0x0203A9C4's value: {}", current_value);

                        // //display x position 220 panel 6, 180 panel 5, 140 panel 4, 100 panel 3, 60 panel 2, 20 panel 1
                        // let current_value = core_ref.raw_read_16(0x0203AA4C, -1);
                        // println!("0x0203AA4C's value: {}", current_value);

                        //0x02036A8C local player x
                        //0x0203AA9C local player y

                        let player1_x = core_ref.raw_read_16(0x0203AA4C, -1);
                        let player1_y = core_ref.raw_read_16(0x0203A9C4, -1);
                        let player2_x = core_ref.raw_read_16(0x02036A8C, -1);
                        let player2_y = core_ref.raw_read_16(0x0203AA9C, -1);

                        //position of player 1
                        let player_grid_positionX1 = core_ref.raw_read_8(0x0203AA9C, -1);
                        let player_grid_positionY1 = core_ref.raw_read_8(0x0203AA9D, -1);
                        // println!("Player grid position1: {}, {}", player_grid_positionX1, player_grid_positionY1);

                        //position of player 2
                        let player_grid_positionX2 = core_ref.raw_read_8(0x020384FA, -1);
                        let player_grid_positionY2 = core_ref.raw_read_8(0x020384FB, -1);
                        // println!("Player grid position2: {}, {}", player_grid_positionX2, player_grid_positionY2);

                        if is_offerer{
                            set_player_game_emotion_state(core_ref.raw_read_8(0x0203CE90, -1) as u16);
                            set_enemy_game_emotion_state(core_ref.raw_read_8(0x0203CE2C, -1) as u16);
                        }
                        else{
                            set_player_game_emotion_state(core_ref.raw_read_8(0x0203CE2C, -1) as u16);
                            set_enemy_game_emotion_state(core_ref.raw_read_8(0x0203CE90, -1) as u16);
                        }

                        if get_player_health_index() == 0 {
                            set_player_position((player1_x, player1_y));
                            set_enemy_position((player2_x, player2_y));

                            set_player_charge(core_ref.raw_read_8(0x0203409D, -1) as u16);
                            set_enemy_charge(core_ref.raw_read_8(0x0203419D, -1) as u16);

                            // println!("First Chip P1: {}", core.raw_read_16(0x203A9DA, -1));
                            set_player_selected_chip(core_ref.raw_read_16(0x203A9DA, -1));
                            // println!("First Chip P2: {}", core.raw_read_16(0x203AAB2, -1));
                            set_enemy_selected_chip(core_ref.raw_read_16(0x203AAB2, -1));

                            set_player_grid_position((player_grid_positionX1 as u16, player_grid_positionY1 as u16));
                            set_enemy_grid_position((player_grid_positionX2 as u16, player_grid_positionY2 as u16));
                            // println!("Player game emotion state: {}", core.raw_read_8(0x0203CE90, -1) as u16);
                            // println!("P2 emotion state: {}", core.raw_read_8(0x0203CE2C, -1) as u16);
                        } else if get_player_health_index() == 1 {
                            set_player_position((player2_x, player2_y));
                            set_enemy_position((player1_x, player1_y));

                            set_player_charge(core_ref.raw_read_8(0x0203419D, -1) as u16);
                            set_enemy_charge(core_ref.raw_read_8(0x0203409D, -1) as u16);

                            // println!("First Chip P2: {}", core.raw_read_16(0x203A9DA, -1));
                            set_enemy_selected_chip(core_ref.raw_read_16(0x203A9DA, -1));
                            // println!("First Chip P1: {}", core.raw_read_16(0x203AAB2, -1));
                            set_player_selected_chip(core_ref.raw_read_16(0x203AAB2, -1));




                            set_player_grid_position((player_grid_positionX2 as u16, player_grid_positionY2 as u16));
                            set_enemy_grid_position((player_grid_positionX1 as u16, player_grid_positionY1 as u16));
                            // println!("Player game emotion state: {}", core.raw_read_8(0x0203CE2C, -1) as u16);
                            // println!("P2 emotion state: {}", core.raw_read_8(0x0203CE90, -1) as u16);
                        } else {
                            set_player_position((0, 0));
                        }
                    } else {
                        println!("Health addresses not properly set.");
                    }
                    // Search for health values that start at 1250
                    // Search for health values that start at 1250
                    // let health_addresses = search_all_health_values(core_ref, 1250);

                    // // Access the global health addresses map without using unwrap()
                    // let mut health_map = HEALTH_ADDRESSES.lock();

                    // // Update the tracked health addresses with new potential addresses
                    // for addr in &health_addresses {
                    //     // Initialize address in the map if not already present
                    //     health_map.entry(*addr).or_insert(1250);
                    // }

                    // // Check the current values of all tracked addresses
                    // health_map.retain(|&addr, last_value| {
                    //     let current_value = core_ref.raw_read_16(addr, -1);

                    //     // Check if the health value changed as expected
                    //     if current_value != *last_value {
                    //         println!(
                    //             "Health value change detected at address 0x{:08X}: {} -> {}",
                    //             addr, last_value, current_value
                    //         );

                    //         // Update the last known value for this address
                    //         *last_value = current_value;

                    //         // Return true to keep this address in the list if the current value is plausible
                    //         // Here, we check if it's a reasonable health value (e.g., less than 1250)
                    //         return current_value <= 1250;
                    //     }

                    //     // Retain address if no change occurred, waiting for a potential change
                    //     true
                    // });

                    // // Output remaining potential health addresses
                    // println!("Potential health addresses after filtering:");
                    // for (addr, value) in &*health_map {
                    //     println!("Address: 0x{:08X}, Current Value: {}", addr, value);
                    // }
                }
            }
        });
        println!("Replayer created, loading setup data");
        // Load own setup
        let own_wram_data = replay.local_state.wram();
        let own_save = match game.save_from_wram(own_wram_data) {
            Ok(save) => save,
            Err(e) => {
                log::error!("failed to load save from wram: {:?}", e);
                return Err(e.into());
            }
        };

        let own_assets = game.load_rom_assets(rom, own_wram_data, &rom::Overrides::default())?;

        let own_setup = Some(Setup {
            game_lang: crate::game::region_to_language(game.gamedb_entry().region),
            save: own_save,
            assets: own_assets,
        });

        // Load opponent setup
        let opponent_setup = if let Some(remote_side) = replay.metadata.remote_side.as_ref() {
            if let Some(remote_game_info) = remote_side.game_info.as_ref() {
                if let Some(remote_game) = game::find_by_family_and_variant(
                    remote_game_info.rom_family.as_str(),
                    remote_game_info.rom_variant as u8,
                ) {
                    // Construct the key for roms
                    let key = (remote_game_info.rom_family.clone(), remote_game_info.rom_variant as u8);

                    // Load the remote ROM
                    let remote_rom = roms
                        .get(&key)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("Remote ROM not found for game {:?}", remote_game_info))?;

                    // Proceed with loading the opponent's save and assets
                    let opponent_wram_data = replay.remote_state.wram();
                    let opponent_save = remote_game.save_from_wram(opponent_wram_data)?;

                    let opponent_assets =
                        remote_game.load_rom_assets(&remote_rom, opponent_wram_data, &rom::Overrides::default())?;

                    Some(Setup {
                        game_lang: crate::game::region_to_language(remote_game.gamedb_entry().region),
                        save: opponent_save,
                        assets: opponent_assets,
                    })
                } else {
                    log::error!("Could not find remote game");
                    None
                }
            } else {
                log::error!("No remote game info");
                None
            }
        } else {
            log::error!("No remote side info");
            None
        };

        //get setups and set the global chip folder and print navi cust data
        // After initializing `own_setup` and `opponent_setup`
        let own_navi_cust_ids = if let Some(ref setup) = own_setup {
            extract_navi_cust_ids(&*setup.save, setup.assets.as_ref())
        } else {
            Vec::new()
        };

        let opponent_navi_cust_ids = if let Some(ref setup) = opponent_setup {
            extract_navi_cust_ids(&*setup.save, setup.assets.as_ref())
        } else {
            Vec::new()
        };

        // Log the Navi Cust IDs
        println!("Local Player Navi Cust IDs: {:?}", own_navi_cust_ids);
        println!("Opponent Player Navi Cust IDs: {:?}", opponent_navi_cust_ids);

        set_player_navi_cust_parts(own_navi_cust_ids);
        set_enemy_navi_cust_parts(opponent_navi_cust_ids);

        // **Extract and Print Chips**
        let own_chips = if let Some(ref setup) = own_setup {
            extract_chips(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        let opponent_chips = if let Some(ref setup) = opponent_setup {
            extract_chips(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        // **Extract and Print Chips**
        let own_tag_chips = if let Some(ref setup) = own_setup {
            extract_tag_indices(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        let opponent_tag_chips = if let Some(ref setup) = opponent_setup {
            extract_tag_indices(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(Vec::new())
        };

        // Extract regular chip index
        let own_reg_chip_index = if let Some(ref setup) = own_setup {
            extract_reg_index(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(60)
        };

        let opponent_reg_chip_index = if let Some(ref setup) = opponent_setup {
            extract_reg_index(&*setup.save, setup.assets.as_ref())
        } else {
            Ok(60)
        };

        let own_reg_chip_index = own_reg_chip_index.unwrap_or(60);
        set_player_reg_chip(own_reg_chip_index as u16);
        let opponent_reg_chip_index = opponent_reg_chip_index.unwrap_or(60);
        set_enemy_reg_chip(opponent_reg_chip_index as u16);

        // Log the Chips
        println!("Local Player Chips:");
        if let Ok(chips) = own_chips {
            let mut index = 0;
            for chip in chips {
                // println!("  ID: {}, Code: {}", chip.id, chip.code);
                if let Err(e) = set_player_chip_folder(index, chip.id as u16) {
                    eprintln!("Failed to set player chip folder: {:?}", e);
                }
                match set_player_code_folder(index, chip.code as u16) {
                    Ok(_) => {}
                    Err(e) => eprintln!("Failed to set player code folder: {:?}", e),
                }
                index += 1;
            }
        }

        println!("Opponent Player Chips:");
        if let Ok(chips) = opponent_chips {
            let mut index = 0;
            for chip in chips {
                if let Err(e) = set_enemy_chip_folder(index, chip.id as u16) {
                    eprintln!("Failed to set enemy chip folder: {:?}", e);
                }
                if let Err(e) = set_enemy_code_folder(index, chip.code as u16) {
                    eprintln!("Failed to set enemy code folder: {:?}", e);
                }
                // println!("  ID: {}, Code: {}", chip.id, chip.code);
                index += 1;
            }
        }

        // Log the Tag Chips
        println!("Local Player Tag Chips:");
        if let Ok(tag_chips) = own_tag_chips {
            let mut index = 0;
            for tag_chip in tag_chips {
                // println!("Tag Chip: {}", tag_chip);
                set_player_tag_folder(index, tag_chip as u16);
                index += 1;
            }
        }
        println!("Opponent Player Tag Chips:");
        if let Ok(tag_chips) = opponent_tag_chips {
            let mut index = 0;
            for tag_chip in tag_chips {
                // println!("Tag Chip: {}", tag_chip);
                set_enemy_tag_folder(index, tag_chip as u16);
                index += 1;
            }
        }
        //opponent_setup = None;

        Ok(Session {
            start_time: std::time::SystemTime::now(),
            game_info: GameInfo { game, patch },
            vbuf,
            _audio_binding: audio_binding,
            thread,
            joyflags: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            mode: Mode::Replayer,
            completion_token,
            pause_on_next_frame,
            own_setup: None,
            opponent_setup: None,
        })
    }

    pub fn completed(&self) -> bool {
        self.completion_token.is_complete()
    }

    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    pub fn set_paused(&self, pause: bool) {
        let handle = self.thread.handle();
        if pause {
            handle.pause();
        } else {
            handle.unpause();
        }
    }

    pub fn is_paused(&self) -> bool {
        let handle = self.thread.handle();
        handle.is_paused()
    }

    pub fn frame_step(&self) {
        self.pause_on_next_frame
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let handle = self.thread.handle();
        handle.unpause();
    }

    pub fn set_fps_target(&self, fps: f32) {
        let handle = self.thread.handle();
        let audio_guard = handle.lock_audio();
        audio_guard.sync_mut().set_fps_target(fps);
    }

    pub fn fps_target(&self) -> f32 {
        let handle = self.thread.handle();
        let audio_guard = handle.lock_audio();
        audio_guard.sync().fps_target()
    }

    pub fn set_master_volume(&self, volume: i32) {
        let handle = self.thread.handle();
        let mut audio_guard = handle.lock_audio();
        audio_guard.core_mut().gba_mut().set_master_volume(volume);
    }

    pub fn has_crashed(&self) -> Option<mgba::thread::Handle> {
        let handle = self.thread.handle();
        if handle.has_crashed() {
            Some(handle)
        } else {
            None
        }
    }

    pub fn lock_vbuf(&self) -> parking_lot::MutexGuard<Vec<u8>> {
        self.vbuf.lock()
    }

    pub fn thread_handle(&self) -> mgba::thread::Handle {
        self.thread.handle()
    }

    pub fn set_joyflags(&self, joyflags: u32) {
        self.joyflags.store(joyflags, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn game_info(&self) -> &GameInfo {
        &self.game_info
    }

    pub fn start_time(&self) -> std::time::SystemTime {
        self.start_time
    }

    pub fn opponent_setup(&self) -> &Option<Setup> {
        &self.opponent_setup
    }

    pub fn own_setup(&self) -> &Option<Setup> {
        &self.own_setup
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Mode::PvP(pvp) = &mut self.mode {
            pvp.cancellation_token.cancel();
        }
    }
}
