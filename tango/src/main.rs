#![windows_subsystem = "windows"]

use global::{get_all_player_chip_folders, get_frame_count, get_player_charge, get_selected_menu_index};
use std::io::Write;
use std::sync::Arc;
use tango_pvp::replay;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;

use clap::Parser;

#[macro_use]
extern crate lazy_static;

mod audio;
mod config;
mod controller;
mod discord;
mod game;
mod graphics;
mod gui;
mod i18n;
mod input;
mod keyboard;
mod net;
mod patch;
mod randomcode;
mod rom;
mod save;
mod scanner;
mod session;
mod stats;
mod sync;
mod updater;
mod version;
mod video;

use fluent_templates::Loader;
use keyboard::Key;

mod global; // Include the global module

use crate::global::{
    add_punishment, add_reward, clear_local_input, clear_punishments, clear_rewards, get_all_chip_codes,
    get_all_chip_slots, get_all_enemy_chip_folders, get_all_enemy_code_folders, get_all_enemy_tag_folders,
    get_all_player_code_folders, get_all_player_tag_folders, get_all_selected_chip_indices, get_beast_out_selectable,
    get_chip_code, get_chip_count_visible, get_chip_selected_count, get_chip_slot, get_enemy_charge,
    get_enemy_emotion_state, get_enemy_game_emotion_state, get_enemy_health, get_enemy_position, get_enemy_reg_chip,
    get_enemy_selected_chip, get_inside_cross_window, get_is_player_inside_window, get_local_input,
    get_player_emotion_state, get_player_game_emotion_state, get_player_health, get_player_position,
    get_player_reg_chip, get_player_selected_chip, get_punishments, get_rewards, get_screen_image,
    get_selected_chip_index, get_selected_cross_index, get_winner, RewardPunishment, SCREEN_IMAGE,
};
use crate::global::{PUNISHMENTS, REWARDS}; // Import the global variables

use base64::encode; // Add base64 for encoding images as strings

const TANGO_CHILD_ENV_VAR: &str = "TANGO_CHILD";

#[derive(Debug)]
struct Args {
    init_link_code: String,
    ai_model: String,
    rom: String,
    save: String,
    port: u16,                   // Added port number
    replay_path: Option<String>, // Add replay path argument
}

impl Args {
    fn from_env() -> Result<Self, anyhow::Error> {
        Ok(Self {
            init_link_code: std::env::var("INIT_LINK_CODE")?,
            ai_model: std::env::var("AI_MODEL_PATH")?,
            rom: std::env::var("ROM_PATH")?,
            save: std::env::var("SAVE_PATH")?,
            port: std::env::var("PORT")?.parse::<u16>()?, // Read port from environment variable
            replay_path: std::env::var("REPLAY_PATH").ok(), // Read replay path from environment variable
        })
    }
}
enum UserEvent {
    RequestRepaint,
}

use lazy_static::lazy_static;

fn main() -> Result<(), anyhow::Error> {
    std::env::set_var("RUST_BACKTRACE", "FULL");

    let args = Args::from_env()?; // Read arguments from environment variables
    println!("Parsed arguments from env: {:?}", args);

    // Check if the REPLAY_PATH environment variable is set and store it globally
    if let Ok(replay_path) = std::env::var("REPLAY_PATH") {
        global::set_replay_path(replay_path);
    }

    let config = config::Config::load_or_create()?;
    config.ensure_dirs()?;

    env_logger::Builder::from_default_env()
        .filter(Some("tango"), log::LevelFilter::Info)
        .filter(Some("datachannel"), log::LevelFilter::Info)
        .filter(Some("mgba"), log::LevelFilter::Info)
        .init();

    log::info!("welcome to tango {}!", version::current());

    if std::env::var(TANGO_CHILD_ENV_VAR).unwrap_or_default() == "1" {
        return child_main(config, args);
    }

    let log_filename = format!(
        "{}.log",
        time::OffsetDateTime::from(std::time::SystemTime::now())
            .format(time::macros::format_description!(
                "[year padding:zero][month padding:zero repr:numerical][day padding:zero][hour padding:zero][minute padding:zero][second padding:zero]"
            ))
            .expect("format time"),
    );

    let log_path = config.logs_path().join(log_filename);
    log::info!("logging to: {}", log_path.display());

    let mut log_file = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => {
            rfd::MessageDialog::new()
                //.set_title(&i18n::LOCALES.lookup(&config.language, "window-title").unwrap())
                .set_description(
                    &i18n::LOCALES
                        .lookup_with_args(
                            &config.language,
                            "crash-no-log",
                            &std::collections::HashMap::from([("error", format!("{:?}", e).into())]),
                        )
                        .unwrap(),
                )
                .set_level(rfd::MessageLevel::Error)
                .show();
            return Err(e.into());
        }
    };

    let status = std::process::Command::new(std::env::current_exe()?)
        .args(std::env::args_os().skip(1).collect::<Vec<std::ffi::OsString>>())
        .env(TANGO_CHILD_ENV_VAR, "1")
        .stderr(log_file.try_clone()?)
        .spawn()?
        .wait()?;

    writeln!(&mut log_file, "exit status: {:?}", status)?;

    if !status.success() {
        rfd::MessageDialog::new()
            //.set_title(&i18n::LOCALES.lookup(&config.language, "window-title").unwrap())
            .set_description(
                &i18n::LOCALES
                    .lookup_with_args(
                        &config.language,
                        "crash",
                        &std::collections::HashMap::from([("path", format!("{}", log_path.display()).into())]),
                    )
                    .unwrap(),
            )
            .set_level(rfd::MessageLevel::Error)
            .show();
    }

    if let Some(code) = status.code() {
        std::process::exit(code);
    }

    Ok(())
}

fn child_main(mut config: config::Config, args: Args) -> Result<(), anyhow::Error> {
    // Use the init_link_code from args
    let init_link_code = args.init_link_code;
    let rom_path = args.rom;
    let save_path = args.save;
    let port = args.port;

    let replay_path = args.replay_path.unwrap_or_default();

    // Create a separate runtime for asynchronous tasks
    let rt = Runtime::new()?;

    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<InputCommand>(); // For receiving input commands
    let (output_tx, mut output_rx) = mpsc::unbounded_channel::<OutputMessage>(); // For sending messages back to Python
    let output_tx = Arc::new(Mutex::new(Some(output_tx))); // Wrap in Arc<Mutex<>> for sharing between threads

    // Run the async task within this runtime
    rt.spawn(setup_tcp_listener(port, input_tx.clone(), output_tx.clone()));

    println!("Using init_link_code: {}", init_link_code);

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let _enter_guard = rt.enter();

    let show_update_info = config.last_version != version::current();
    config.last_version = version::current();

    config.save()?;
    let config = std::sync::Arc::new(parking_lot::RwLock::new(config));

    let updater_path = config::get_updater_path().unwrap();
    let _ = std::fs::create_dir_all(&updater_path);
    let mut updater = updater::Updater::new(&updater_path, config.clone());
    updater.set_enabled(config.read().enable_updater);

    let sdl = sdl2::init().unwrap();
    let game_controller = sdl.game_controller().unwrap();

    let event_loop = winit::event_loop::EventLoopBuilder::with_user_event().build().unwrap();
    let mut sdl_event_loop = sdl.event_pump().unwrap();

    let icon = image::load_from_memory(include_bytes!("icon.png"))?;
    let icon_width = icon.width();
    let icon_height = icon.height();
    let window_title = format!("{}", args.port);

    let window_builder = winit::window::WindowBuilder::new()
        .with_title(&window_title) // Set the title to the port number
        //.with_title(i18n::LOCALES.lookup(&config.read().language, "window-title").unwrap())
        .with_window_icon(Some(winit::window::Icon::from_rgba(
            icon.into_bytes(),
            icon_width,
            icon_height,
        )?))
        .with_inner_size(config.read().window_size)
        .with_min_inner_size(winit::dpi::PhysicalSize::new(
            mgba::gba::SCREEN_WIDTH,
            mgba::gba::SCREEN_HEIGHT,
        ))
        .with_fullscreen(if config.read().full_screen {
            Some(winit::window::Fullscreen::Borderless(None))
        } else {
            None
        });

    let mut gfx_backend: Box<dyn graphics::Backend> = match config.read().graphics_backend {
        #[cfg(feature = "glutin")]
        config::GraphicsBackend::Glutin => Box::new(graphics::glutin::Backend::new(window_builder, &event_loop)?),
        #[cfg(feature = "wgpu")]
        config::GraphicsBackend::Wgpu => Box::new(graphics::wgpu::Backend::new(window_builder, &event_loop)?),
    };
    gfx_backend.set_ui_scale(config.read().ui_scale_percent as f32 / 100.0);
    gfx_backend.run(&mut |_, _| {});
    gfx_backend.paint();

    let egui_ctx = gfx_backend.egui_ctx();
    egui_extras::install_image_loaders(egui_ctx);
    egui_ctx.set_request_repaint_callback({
        let el_proxy = parking_lot::Mutex::new(event_loop.create_proxy());
        move |_| {
            let _ = el_proxy.lock().send_event(UserEvent::RequestRepaint);
        }
    });
    updater.set_ui_callback({
        let egui_ctx = egui_ctx.clone();
        Some(Box::new(move || {
            egui_ctx.request_repaint();
        }))
    });

    let mut audio_binder = audio::LateBinder::new();
    let audio_backend: Box<dyn audio::Backend> = match config.read().audio_backend {
        #[cfg(feature = "cpal")]
        config::AudioBackend::Cpal => Box::new(audio::cpal::Backend::new(audio_binder.clone())?),
        #[cfg(feature = "sdl2-audio")]
        config::AudioBackend::Sdl2 => Box::new(audio::sdl2::Backend::new(&sdl, audio_binder.clone())?),
    };
    audio_binder.set_sample_rate(audio_backend.sample_rate());

    let fps_counter = std::sync::Arc::new(parking_lot::Mutex::new(stats::Counter::new(30)));
    let emu_tps_counter = std::sync::Arc::new(parking_lot::Mutex::new(stats::Counter::new(10)));

    let mut input_state = input::State::new();

    let mut controllers: std::collections::HashMap<u32, sdl2::controller::GameController> =
        std::collections::HashMap::new();
    // Preemptively enumerate controllers.
    for which in 0..game_controller.num_joysticks().unwrap() {
        if !game_controller.is_game_controller(which) {
            continue;
        }
        match game_controller.open(which) {
            Ok(controller) => {
                log::info!("controller added: {}", controller.name());
                controllers.insert(which, controller);
            }
            Err(e) => {
                log::info!("failed to add controller: {}", e);
            }
        }
    }

    let discord_client = discord::Client::new();

    let roms_scanner = scanner::Scanner::new();
    let saves_scanner = scanner::Scanner::new();
    let patches_scanner = scanner::Scanner::new();
    {
        let roms_path = config.read().roms_path();
        let saves_path = config.read().saves_path();
        let patches_path = config.read().patches_path();
        roms_scanner.rescan(move || Some(game::scan_roms(&roms_path)));
        saves_scanner.rescan(move || Some(save::scan_saves(&saves_path)));
        patches_scanner.rescan(move || Some(patch::scan(&patches_path).unwrap_or_default()));
    }

    let mut state = gui::State::new(
        egui_ctx,
        show_update_info,
        config.clone(),
        discord_client,
        audio_binder.clone(),
        fps_counter.clone(),
        emu_tps_counter.clone(),
        roms_scanner.clone(),
        saves_scanner.clone(),
        patches_scanner.clone(),
        init_link_code,
        rom_path,
        save_path,
        port,
    )?;

    let mut patch_autoupdater = patch::Autoupdater::new(config.clone(), patches_scanner.clone());
    patch_autoupdater.set_enabled(config.read().enable_patch_autoupdate);

    let mut last_config_dirty_time = None;
    event_loop.run(move |event, window_target| {
        let mut next_config = config.read().clone();
        let old_config = next_config.clone();

        let mut redraw = || {
            let repaint_after = gfx_backend
                .run(&mut (|window, ctx| gui::show(ctx, &mut next_config, window, &input_state, &mut state, &updater)));

            if repaint_after.is_zero() {
                gfx_backend.window().request_redraw();
                window_target.set_control_flow(winit::event_loop::ControlFlow::Poll);
            } else if let Some(repaint_after_instant) = std::time::Instant::now().checked_add(repaint_after) {
                window_target.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(repaint_after_instant));
            } else {
                window_target.set_control_flow(winit::event_loop::ControlFlow::Wait);
            }

            gfx_backend.paint();
            fps_counter.lock().mark();
        };

        match event {
            winit::event::Event::WindowEvent {
                event: window_event, ..
            } => {
                match window_event {
                    winit::event::WindowEvent::RedrawRequested if !cfg!(windows) => redraw(),
                    winit::event::WindowEvent::MouseInput { .. } | winit::event::WindowEvent::CursorMoved { .. } => {
                        state.last_mouse_motion_time = Some(std::time::Instant::now());
                        if state.steal_input.is_none() {
                            let _ = gfx_backend.on_window_event(&window_event);
                        }
                    }
                    winit::event::WindowEvent::KeyboardInput {
                        event:
                            winit::event::KeyEvent {
                                physical_key: winit::keyboard::PhysicalKey::Code(winit_key),
                                state: element_state,
                                ..
                            },
                        ..
                    } => {
                        if let Some(key) = Key::resolve(winit_key) {
                            match element_state {
                                winit::event::ElementState::Pressed => {
                                    if let Some(steal_input) = state.steal_input.take() {
                                        steal_input.run_callback(
                                            input::PhysicalInput::Key(key),
                                            &mut next_config.input_mapping,
                                        );
                                    } else if !gfx_backend.on_window_event(&window_event).consumed {
                                        input_state.handle_key_down(key);
                                    } else {
                                        input_state.clear_keys();
                                    }
                                }
                                winit::event::ElementState::Released => {
                                    if !gfx_backend.on_window_event(&window_event).consumed {
                                        input_state.handle_key_up(key);
                                    } else {
                                        input_state.clear_keys();
                                    }
                                }
                            }
                        }
                    }
                    window_event => {
                        let _ = gfx_backend.on_window_event(&window_event);
                        match window_event {
                            // winit::event::WindowEvent::Focused(false) => {
                            //     input_state.clear_keys();
                            // }
                            winit::event::WindowEvent::Occluded(false) => {
                                next_config.full_screen = gfx_backend.window().fullscreen().is_some();
                            }
                            winit::event::WindowEvent::CursorEntered { .. } => {
                                state.last_mouse_motion_time = Some(std::time::Instant::now());
                            }
                            winit::event::WindowEvent::CursorLeft { .. } => {
                                state.last_mouse_motion_time = None;
                            }
                            winit::event::WindowEvent::CloseRequested => {
                                window_target.exit();
                            }
                            _ => {}
                        }
                    }
                };
                gfx_backend.window().request_redraw();
            }
            winit::event::Event::NewEvents(cause) => {
                input_state.digest();
                if let winit::event::StartCause::ResumeTimeReached { .. } = cause {
                    gfx_backend.window().request_redraw();
                }
            }
            winit::event::Event::UserEvent(UserEvent::RequestRepaint) => {
                gfx_backend.window().request_redraw();
            }
            winit::event::Event::AboutToWait => {
                if cfg!(windows) {
                    redraw();
                }

                // Process commands from the Python app
                while let Ok(cmd) = input_rx.try_recv() {
                    // Debug log for received command
                    // println!("Received command from TCP: {:?}", cmd);

                    // Clear input state to release any previously pressed keys
                    input_state.clear_keys();

                    // Simulate keyboard input based on the command
                    match cmd.command_type.as_str() {
                        "key_press" => {
                            // Loop over each bit in the binary string and simulate key presses
                            for (i, bit) in cmd.key.chars().rev().enumerate() {
                                if bit == '1' {
                                    if let Some(key) = map_bit_to_key(i) {
                                        // Simulate a key press event for the active bits
                                        // println!("Simulating key press for bit position: {}", i);
                                        handle_input_event(
                                            &mut input_state,
                                            &mut state,
                                            key,
                                            winit::event::ElementState::Pressed,
                                            &mut next_config,
                                        );
                                    } else {
                                        // println!("Unrecognized key for bit position: {}", i);
                                    }
                                }
                            }
                        }
                        _ => {
                            println!("Unknown command type: {}", cmd.command_type);
                        }
                    }
                }
            }

            _ => {}
        }

        if let Some(session) = state.shared.session.lock().as_mut() {
            session.set_joyflags(next_config.input_mapping.to_mgba_keys(&input_state));
            session.set_master_volume(next_config.volume);
        }

        // Now handle global rewards and punishments instead of session-specific ones
        let rewards = get_rewards(); // Use global function to get rewards
        let punishments = get_punishments(); // Use global function to get punishments

        // if !rewards.is_empty() {
        //     // println!("Global rewards: {:?}", rewards);
        //     if let Some(ref output_tx) = *output_tx.lock() {
        //         for reward in rewards {
        //             let message = OutputMessage {
        //                 event: "reward".to_string(),
        //                 details: format!("damage: {}", reward.damage),
        //             };
        //             // println!("Sending message: {:?}", message);
        //             if let Err(e) = output_tx.send(message) {
        //                 println!("Failed to send reward message: {}", e);
        //             }
        //         }
        //         clear_rewards(); // Clear global rewards after processing
        //     }
        // }

        // if !punishments.is_empty() {
        //     // println!("Global punishments: {:?}", punishments);
        //     if let Some(ref output_tx) = *output_tx.lock() {
        //         for punishment in punishments {
        //             let message = OutputMessage {
        //                 event: "punishment".to_string(),
        //                 details: format!("damage: {}", punishment.damage),
        //             };
        //             if let Err(e) = output_tx.send(message) {
        //                 println!("Failed to send punishment message: {}", e);
        //             }
        //         }
        //         clear_punishments(); // Clear global punishments after processing
        //     }
        // }

        if let Some(player_won) = get_winner() {
            // Send a winner message to the Python script
            if let Some(ref output_tx) = *output_tx.lock() {
                let message = OutputMessage {
                    event: "winner".to_string(),
                    details: format!("{}", player_won), // Sends "true" if the player won, "false" otherwise
                };
                if let Err(e) = output_tx.send(message) {
                    println!("Failed to send winner message: {}", e);
                }
            }

            // Exit the application after sending the winner message
            std::process::exit(0);
        }

        // Define the directory where training data will be saved
        // only do if replay path is set
        // if !replay_path.is_empty() {
        //     //split replay path on the last /
        //     let filename_path = replay_path.split("/").last().unwrap();
        //     //remove everything after .
        //     let path = format!("/home/lee/TANGO/training_data/{}", filename_path.split(".").next().unwrap());
        //     let training_data_dir = Path::new(&path);

        //     // Retrieve the screen image
        //     // only save the screen image once every 60 frames
        //     let current_frame_count = get_frame_count();
        //     if current_frame_count % 2 == 0 {
        //         if let Some(image) = get_screen_image() {
        //             // Convert Color32 slice to raw bytes (RGBA)
        //             let rgba_bytes: Vec<u8> = image.pixels.iter().flat_map(|pixel| {
        //                 vec![pixel.r(), pixel.g(), pixel.b(), pixel.a()]
        //             }).collect();

        //             // Encode image to PNG format
        //             let mut png_data = Vec::new();
        //             let encoder = PngEncoder::new(&mut png_data);
        //             encoder.write_image(
        //                 &rgba_bytes,
        //                 image.size[0] as u32,
        //                 image.size[1] as u32,
        //                 ColorType::Rgba8.into(),
        //             ).expect("Failed to encode image");

        //             // Save the game state locally
        //             let inputString = match get_local_input() {
        //                 Some(input) => format!("{:016b}", input),
        //                 None => "0000000000000000".to_string(), // or handle None case appropriately
        //             };
        //             // println!("Local input: {:?}", inputString);
        //             if let Err(e) = save_game_state(
        //                 &png_data,
        //                 &inputString,
        //                 get_player_health(),
        //                 get_enemy_health(),
        //                 get_player_position(),
        //                 get_enemy_position(),
        //                 get_is_player_inside_window(),
        //                 get_rewards().last().cloned(),
        //                 get_punishments().last().cloned(),
        //                 get_player_charge(),
        //                 get_enemy_charge(),
        //                 training_data_dir,
        //             ) {
        //                 // println!("Failed to save game state: {:?}", e);
        //             }
        //             //clear rewards and punishments
        //             clear_rewards();
        //             clear_punishments();
        //         }
        //     }
        // }

        // Handling local inputs
        let local_inputs = get_local_input();
        if let Some(input) = local_inputs {
            if let Some(ref output_tx) = *output_tx.lock() {
                let input_message = OutputMessage {
                    event: "local_input".to_string(),
                    details: format!("{:?}", input),
                };
                // if let Err(e) = output_tx.send(input_message) {
                //     println!("Failed to send local inputs: {}", e);
                // }
                // clear_local_input(); // Clear inputs after sending
            }
        }

        //save data

        next_config.window_size = gfx_backend
            .window()
            .inner_size()
            .to_logical(gfx_backend.window().scale_factor());

        if next_config != old_config {
            last_config_dirty_time = Some(std::time::Instant::now());
            *config.write() = next_config.clone();
        }

        if last_config_dirty_time
            .map(|t| (std::time::Instant::now() - t) > std::time::Duration::from_secs(1))
            .unwrap_or(false)
        {
            let r = next_config.save();
            log::info!("config flushed: {:?}", r);
            last_config_dirty_time = None;
        }

        gfx_backend.set_ui_scale(next_config.ui_scale_percent as f32 / 100.0);
        patch_autoupdater.set_enabled(next_config.enable_patch_autoupdate);
        updater.set_enabled(next_config.enable_updater);
    })?;

    Ok(())
}
use serde::Serialize;

#[derive(Serialize, Debug)]
struct GameState {
    image_path: String,
    input: String,
    player_health: u16,
    enemy_health: u16,
    player_position: Option<(u16, u16)>,
    enemy_position: Option<(u16, u16)>,
    inside_window: Option<bool>,
    reward: u16,
    punishment: u16,
    player_charge: u16,
    enemy_charge: u16,
}

// Add a function to map Python command keys to physical keys in the game
fn map_key_to_physical_key(key: &str) -> Option<Key> {
    match key.to_lowercase().as_str() {
        "up" => Some(Key::Up),
        "down" => Some(Key::Down),
        "left" => Some(Key::Left),
        "right" => Some(Key::Right),
        "z" => Some(Key::Z),
        "x" => Some(Key::X),
        "a" => Some(Key::A),
        "s" => Some(Key::S),
        "return" => Some(Key::Return),
        _ => None,
    }
}

use anyhow::Result;
use image::codecs::png::PngEncoder;
use image::ColorType;
use serde_json::to_string_pretty;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
/// Saves the game state by writing the screenshot and the corresponding JSON data.
///
/// # Arguments
///
/// * `image_bytes` - A slice of bytes representing the PNG-encoded image.
/// * `input_binary` - A string representing the input binary data.
/// * `player_health` - The player's health value.
/// * `enemy_health` - The enemy's health value.
/// * `player_position` - The player's position as an optional tuple.
/// * `enemy_position` - The enemy's position as an optional tuple.
/// * `inside_window` - Whether the player is inside the window.
/// * `reward` - An optional reward.
/// * `punishment` - An optional punishment.
/// * `training_data_dir` - The directory where the JSON file will be saved.
///
/// # Returns
///
/// * `Result<()>` - Ok if successful, Err otherwise.
///
fn save_game_state(
    image_bytes: &[u8],
    input_binary: &str,
    player_health: u16,
    enemy_health: u16,
    player_position: Option<(u16, u16)>,
    enemy_position: Option<(u16, u16)>,
    inside_window: Option<bool>,
    reward: Option<RewardPunishment>,
    punishment: Option<RewardPunishment>,
    player_charge: u16,
    enemy_charge: u16,
    training_data_dir: &Path,
) -> Result<()> {
    // Convert to absolute path
    // let absolute_path = env::current_dir()?.join(training_data_dir);
    // println!("Saving game state to directory: {:?}", absolute_path);

    // // Ensure the training_data_dir exists
    // fs::create_dir_all(&absolute_path)?;

    // // Create a timestamp for the filename
    // let start = SystemTime::now();
    // let since_the_epoch = start.duration_since(UNIX_EPOCH)?;
    // let timestamp = since_the_epoch.as_millis();

    // // Define the image and JSON filenames
    // let image_filename = format!("{}.png", timestamp);
    // let json_filename = format!("{}.json", timestamp);

    // let image_path = training_data_dir.join(&image_filename);
    // let json_path = training_data_dir.join(&json_filename);

    // // Save the image
    // fs::write(&image_path, image_bytes)?;

    // let reward = reward.map(|reward| reward.damage).unwrap_or(0);
    // let punishment = punishment.map(|punishment| punishment.damage).unwrap_or(0);
    // // Create the GameState instance
    // let game_state = GameState {
    //     image_path: image_path.to_string_lossy().to_string(),
    //     input: input_binary.to_string(),
    //     player_health,
    //     enemy_health,
    //     player_position,
    //     enemy_position,
    //     inside_window,
    //     reward,
    //     punishment,
    //     player_charge,
    //     enemy_charge,
    // };

    // // Serialize the GameState to pretty JSON
    // let json_data = to_string_pretty(&game_state)?;

    // // Write the JSON data to the file
    // fs::write(&json_path, json_data)?;

    Ok(())
}

// Function to map bits to specific keys
fn map_bit_to_key(bit: usize) -> Option<Key> {
    match bit {
        8 => Some(Key::A),      // 0000000100000000
        7 => Some(Key::Down),   // 0000000010000000
        6 => Some(Key::Up),     // 0000000001000000
        5 => Some(Key::Left),   // 0000000000100000
        4 => Some(Key::Right),  // 0000000000010000
        9 => Some(Key::S),      // 0000001000000000
        1 => Some(Key::X),      // 0000000000000010
        0 => Some(Key::Z),      // 0000000000000001
        3 => Some(Key::Return), // 0000000000001000 -> enter
        _ => None,
    }
}

// Use this helper function to handle input consistently

// Function to handle input events consistently
fn handle_input_event(
    input_state: &mut input::State,
    state: &mut gui::State,
    key: Key,
    element_state: winit::event::ElementState,
    next_config: &mut config::Config,
) {
    match element_state {
        winit::event::ElementState::Pressed => {
            if let Some(steal_input) = state.steal_input.take() {
                steal_input.run_callback(input::PhysicalInput::Key(key), &mut next_config.input_mapping);
            } else {
                input_state.handle_key_down(key);
            }
        }
        winit::event::ElementState::Released => {
            input_state.handle_key_up(key);
        }
    }
}
// Modify setup_tcp_listener to accept output_tx
async fn setup_tcp_listener(
    port: u16,
    tx: mpsc::UnboundedSender<InputCommand>,
    output_tx: Arc<Mutex<Option<mpsc::UnboundedSender<OutputMessage>>>>,
) -> Result<(), anyhow::Error> {
    let listener = TcpListener::bind(("127.0.0.1", port)).await?;
    println!("Listening for input events on port {}", port);

    loop {
        match listener.accept().await {
            Ok((socket, _)) => {
                println!("Accepted connection on port {}", port);
                tokio::spawn(handle_tcp_client(socket, tx.clone(), output_tx.clone()));
            }
            Err(e) => {
                println!("Failed to accept connection: {}", e);
            }
        }
    }
}

use parking_lot::Mutex;
use serde::Deserialize;
use tokio::io::AsyncWriteExt; // Add this for sending data back
use tokio::sync::mpsc; // Add this for thread safety

#[derive(Deserialize, Serialize, Debug, Clone)] // Add Clone here
struct InputCommand {
    #[serde(rename = "type")]
    command_type: String,
    key: String,
}

// Define a struct for messages sent back to the Python app
#[derive(Serialize, Debug)]
struct OutputMessage {
    event: String,
    details: String,
}

#[derive(Serialize, Debug)]
struct ScreenImageDetails {
    image: String, // Base64-encoded PNG image
    player_health: u16,
    enemy_health: u16,
    player_position: Option<(u16, u16)>,
    enemy_position: Option<(u16, u16)>,
    inside_window: bool,
    player_charge: u16,
    enemy_charge: u16,
    reward: u16,
    punishment: u16,
    current_input: u16,
    player_chip: u16,
    enemy_chip: u16,
    player_emotion: u16,
    enemy_emotion: u16,
    player_game_emotion: u16,
    enemy_game_emotion: u16,
    selected_menu_index: u16,
    selected_cross_index: u16,
    chip_selected_count: u16,
    chip_visible_count: u16,
    chip_slots: Vec<u16>,
    chip_codes: Vec<u16>,
    selected_chip_indices: Vec<u16>,
    beast_out_selectable: u16,
    inside_cross_window: u16,
    player_chip_folder: Vec<u16>,
    enemy_chip_folder: Vec<u16>,
    player_code_folder: Vec<u16>,
    enemy_code_folder: Vec<u16>,
    player_tag_chips: Vec<u16>,
    enemy_tag_chips: Vec<u16>,
    player_reg_chip: u16,
    enemy_reg_chip: u16,
}
use egui::Color32;
use image::ImageEncoder;

async fn handle_tcp_client(
    mut socket: tokio::net::TcpStream,
    tx: mpsc::UnboundedSender<InputCommand>,
    output_tx: Arc<Mutex<Option<mpsc::UnboundedSender<OutputMessage>>>>,
) {
    // Register the output_tx for sending messages
    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<OutputMessage>();
    *output_tx.lock() = Some(msg_tx);

    let mut buf = vec![0; 8192];
    loop {
        tokio::select! {
            // Reading from the socket
            n = socket.read(&mut buf) => {
                match n {
                    Ok(0) => {
                        // Connection closed
                        break;
                    }
                    Ok(n) => {
                        let data = &buf[..n];
                        if let Ok(command_str) = std::str::from_utf8(data) {
                            for line in command_str.lines() {
                                if let Ok(cmd) = serde_json::from_str::<InputCommand>(line) {
                                    match cmd.command_type.as_str() {
                                        "request_screen" => {
                                            // Handle screen image request
                                            if let Some(image) = get_screen_image() {
                                                // Convert Color32 slice to raw bytes (RGBA)
                                                let rgba_bytes: Vec<u8> = image.pixels.iter().flat_map(|pixel| {
                                                    vec![pixel.r(), pixel.g(), pixel.b(), pixel.a()]
                                                }).collect();

                                                // Encode image to PNG format
                                                let mut png_data = Vec::new();
                                                let encoder = PngEncoder::new(&mut png_data);
                                                encoder.write_image(
                                                    &rgba_bytes,
                                                    image.size[0] as u32,
                                                    image.size[1] as u32,
                                                    ColorType::Rgba8.into(),
                                                ).expect("Failed to encode image");

                                                // Encode PNG data in base64
                                                let encoded_image = encode(png_data);

                                                // Retrieve additional game state data
                                                let player_health = get_player_health();
                                                let enemy_health = get_enemy_health();
                                                let player_position = get_player_position();
                                                let enemy_position = get_enemy_position();
                                                // let inside_window = get_is_player_inside_window();
                                                //if inside window is none return false
                                                let inside_window = get_is_player_inside_window().unwrap_or(false);
                                                // Create ScreenImageDetails
                                                let screen_details = ScreenImageDetails {
                                                    image: encoded_image,
                                                    player_health,
                                                    enemy_health,
                                                    player_position,
                                                    enemy_position,
                                                    inside_window,
                                                    player_charge: get_player_charge(),
                                                    enemy_charge: get_enemy_charge(),
                                                    reward: get_rewards().last().map(|reward| reward.damage).unwrap_or(0),
                                                    punishment: get_punishments().last().map(|punishment| punishment.damage).unwrap_or(0),
                                                    current_input: get_local_input().unwrap_or(0),
                                                    player_chip: get_player_selected_chip(),
                                                    enemy_chip: get_enemy_selected_chip(),
                                                    player_emotion: get_player_emotion_state(),
                                                    enemy_emotion: get_enemy_emotion_state(),
                                                    player_game_emotion: get_player_game_emotion_state(),
                                                    enemy_game_emotion: get_enemy_game_emotion_state(),
                                                    selected_menu_index: get_selected_menu_index(),
                                                    selected_cross_index: get_selected_cross_index(),
                                                    chip_selected_count: get_chip_selected_count(),
                                                    chip_visible_count: get_chip_count_visible(),
                                                    chip_slots: get_all_chip_slots(),
                                                    chip_codes: get_all_chip_codes(),
                                                    selected_chip_indices: get_all_selected_chip_indices(),
                                                    beast_out_selectable: get_beast_out_selectable(),
                                                    inside_cross_window: get_inside_cross_window(),
                                                    player_chip_folder: get_all_player_chip_folders(),
                                                    enemy_chip_folder: get_all_enemy_chip_folders(),
                                                    player_code_folder: get_all_player_code_folders(),
                                                    enemy_code_folder: get_all_enemy_code_folders(),
                                                    player_tag_chips: get_all_player_tag_folders(),
                                                    enemy_tag_chips: get_all_enemy_tag_folders(),
                                                    player_reg_chip: get_player_reg_chip(),
                                                    enemy_reg_chip: get_enemy_reg_chip(),
                                                };
                                                clear_local_input();
                                                clear_rewards();
                                                clear_punishments();

                                                // Serialize ScreenImageDetails to JSON
                                                let details_json = serde_json::to_string(&screen_details)
                                                    .expect("Failed to serialize screen details");

                                                // Create OutputMessage
                                                let response = OutputMessage {
                                                    event: "screen_image".to_string(),
                                                    details: details_json,
                                                };

                                                //exit app if player or enemy health is 0
                                                if player_health == 0 && enemy_health != 0
                                                || enemy_health == 0 && player_health != 0 {
                                                    println!("Game Over");
                                                    std::process::exit(0);
                                                }

                                                // Send the response back to Python
                                                if let Err(e) = send_message_to_python(&mut socket, &response).await {
                                                    println!("Failed to send screen image: {}", e);
                                                }
                                            } else {
                                                println!("No screen image available.");
                                            }
                                        }
                                        _ => {
                                            // Handle other commands or send acknowledgment
                                            if tx.send(cmd.clone()).is_err() {
                                                // println!("Failed to send input command to event loop");
                                            }
                                            let response = OutputMessage {
                                                event: "command_received".to_string(),
                                                details: format!("Processed command: {:?}", cmd),
                                            };
                                            if let Err(e) = send_message_to_python(&mut socket, &response).await {
                                                println!("Failed to send message to Python: {}", e);
                                            }
                                        }
                                    }
                                } else {
                                    println!("Failed to parse input command");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to read from socket: {}", e);
                        break;
                    }
                }
            }

            // Reading from msg_rx
            Some(message) = msg_rx.recv() => {
                if let Err(e) = send_message_to_python(&mut socket, &message).await {
                    println!("Failed to send message to Python: {}", e);
                }
            }
            else => {
                // All senders have been dropped
                break;
            }
        }
    }
}

// Function to send messages back to the Python app
async fn send_message_to_python(
    socket: &mut tokio::net::TcpStream,
    message: &OutputMessage,
) -> Result<(), Box<dyn std::error::Error>> {
    let message_json = serde_json::to_string(message)?;
    socket.write_all(message_json.as_bytes()).await?;
    socket.write_all(b"\n").await?;

    //clear rewards
    // clear_rewards();
    // clear_punishments();

    Ok(())
}
