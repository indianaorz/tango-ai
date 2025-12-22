#![windows_subsystem = "windows"]

use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::io::Write; 

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

use keyboard::Key;
use fluent_templates::Loader;

mod global;

use tango_dataview::save::{NaviView, Save, Chip};
use tango_dataview::rom::Assets;

// Import all global functions
use crate::global::{
    add_punishment, add_reward, clear_local_input, clear_punishments, clear_rewards, get_all_chip_codes,
    get_all_chip_slots, get_all_enemy_chip_folders, get_all_enemy_code_folders, get_all_enemy_tag_folders,
    get_all_player_code_folders, get_all_player_tag_folders, get_all_selected_chip_indices, get_beast_out_selectable,
    get_chip_count_visible, get_chip_selected_count, get_enemy_charge,
    get_enemy_emotion_state, get_enemy_game_emotion_state, get_enemy_health, get_enemy_position, get_enemy_reg_chip,
    get_enemy_selected_chip, get_inside_cross_window, get_is_player_inside_window, get_local_input,
    get_player_emotion_state, get_player_game_emotion_state, get_player_health, get_player_position,
    get_player_reg_chip, get_player_selected_chip, get_punishments, get_rewards, get_screen_image,
    get_all_grid_owner_states, get_all_grid_states, get_player_grid_position, get_enemy_grid_position, get_is_offerer,
    get_cust_gage,
    get_selected_chip_index, get_selected_cross_index, get_winner, RewardPunishment,
    get_player_charge, get_selected_menu_index, get_all_player_chip_folders, 
    get_player_navi_cust_parts, get_enemy_navi_cust_parts
};

use base64::encode;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use image::codecs::png::PngEncoder;
use image::ImageEncoder; 
use image::ColorType;

const TANGO_CHILD_ENV_VAR: &str = "TANGO_CHILD";

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    RequestRepaint,
}

// --- CLI DEFINITIONS ---

#[derive(Parser, Debug)]
#[command(name = "tango")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Export a replay to video + JSONL inputs
    Export {
        /// Path to the replay file
        replay_path: PathBuf,

        /// Output path for the video (jsonl will be created alongside)
        #[arg(long)]
        output_path: PathBuf,

        /// Path to the ROM file (e.g., bn6.gba). Defaults to 'bn6_gregar.gba' if not set.
        #[arg(long, default_value = "bn6_gregar.gba")]
        rom_path: PathBuf,
    },
}

#[derive(Debug)]
struct EnvArgs {
    init_link_code: String,
    ai_model: String,
    rom: String,
    save: String,
    port: u16,
    replay_path: Option<String>,
}

impl EnvArgs {
    fn from_env() -> Result<Self, anyhow::Error> {
        Ok(Self {
            init_link_code: std::env::var("INIT_LINK_CODE")?,
            ai_model: std::env::var("AI_MODEL_PATH")?,
            rom: std::env::var("ROM_PATH")?,
            save: std::env::var("SAVE_PATH")?,
            port: std::env::var("PORT")?.parse::<u16>()?,
            replay_path: std::env::var("REPLAY_PATH").ok(),
        })
    }
}

// --- STATIC DATA HELPERS ---

#[derive(serde::Serialize)]
struct StaticReplayData {
    player_navi_cust: Vec<usize>,
    enemy_navi_cust: Vec<usize>,
    player_folder_ids: Vec<u16>,
    player_folder_codes: Vec<u16>,
    enemy_folder_ids: Vec<u16>,
    enemy_folder_codes: Vec<u16>,
    player_reg_chip: u16,
    enemy_reg_chip: u16,
    player_tags: Vec<u16>,
    enemy_tags: Vec<u16>,
}

fn extract_navi_cust_ids(save: &dyn Save, _assets: &dyn Assets) -> Vec<usize> {
    if let Some(navi_view) = save.view_navi() {
        match navi_view {
            NaviView::Navicust(navicust_view) => {
                let size = navicust_view.size();
                let mut ids = Vec::new();
                for i in 0..(size[0] * size[1]) {
                    if let Some(part) = navicust_view.navicust_part(i) {
                        ids.push(part.id);
                    }
                }
                ids
            }
            NaviView::LinkNavi(_) => Vec::new(),
        }
    } else {
        Vec::new()
    }
}

fn extract_chips(save: &dyn Save) -> Vec<Chip> {
    let mut chips = Vec::new();
    if let Some(chips_view) = save.view_chips() {
        let folder_idx = chips_view.equipped_folder_index();
        for i in 0..30 {
            if let Some(chip) = chips_view.chip(folder_idx, i) {
                chips.push(chip);
            }
        }
    }
    chips
}

fn extract_tags(save: &dyn Save) -> Vec<usize> {
    let mut tags = Vec::new();
    if let Some(chips_view) = save.view_chips() {
        let folder_idx = chips_view.equipped_folder_index();
        if let Some(indices) = chips_view.tag_chip_indexes(folder_idx) {
            for &idx in &indices { tags.push(idx); }
        }
    }
    tags
}

fn extract_reg(save: &dyn Save) -> usize {
    if let Some(chips_view) = save.view_chips() {
        let folder_idx = chips_view.equipped_folder_index();
        if let Some(idx) = chips_view.regular_chip_index(folder_idx) {
            return idx;
        }
    }
    255 
}

// --- MAIN ENTRY POINT ---

fn main() -> Result<(), anyhow::Error> {
    std::env::set_var("RUST_BACKTRACE", "FULL");

    env_logger::Builder::from_default_env()
        .filter(Some("tango"), log::LevelFilter::Info)
        .filter(Some("datachannel"), log::LevelFilter::Info)
        .filter(Some("mgba"), log::LevelFilter::Info)
        .filter(Some("tango_pvp"), log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Export { replay_path, output_path, rom_path }) => {
            log::info!("Starting Export Mode...");
            let rt = Runtime::new()?;
            rt.block_on(run_export(replay_path, output_path, rom_path))?;
            Ok(())
        }
        
        None => {
            log::info!("welcome to tango {}!", version::current());

            if std::env::var("INIT_LINK_CODE").is_err() && std::env::var(TANGO_CHILD_ENV_VAR).is_err() {
                 println!("Usage: tango export <REPLAY> --output-path <OUT> --rom-path <ROM>");
                 return Ok(()); 
            }

            let config = config::Config::load_or_create()?;
            config.ensure_dirs()?;

            if std::env::var(TANGO_CHILD_ENV_VAR).unwrap_or_default() == "1" {
                let args = EnvArgs::from_env()?; 
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
    }
}

// --- EXPORT HELPER ---

async fn run_export(replay_path: PathBuf, output_path: PathBuf, rom_path: PathBuf) -> Result<(), anyhow::Error> {
    let mut f = std::fs::File::open(&replay_path)?;
    let replay = tango_pvp::replay::Replay::decode(&mut f)?;

    let rom = std::fs::read(&rom_path)
        .map_err(|e| anyhow::anyhow!("Failed to read ROM at {:?}: {}", rom_path, e))?;

    let detected_game = tango_gamedb::detect(&rom)
        .ok_or(anyhow::anyhow!("ROM detection failed."))?;
    
// --- [INJECTED STATIC DATA EXTRACTION] ---
    // [FIX] Convert enum to string correctly
    if let Some(game_impl) = game::find_by_family_and_variant(
        detected_game.family_and_variant.0.to_string().as_str(),
        detected_game.family_and_variant.1
    ) {
        println!("🎮 Static Data Extraction: Detected {:?}", detected_game.family_and_variant.0);
        
        // Local Player
        let wram = replay.local_state.wram();
        let mut p_navi = Vec::new();
        let mut p_ids = Vec::new();
        let mut p_codes = Vec::new();
        let mut p_tags = Vec::new();
        let mut p_reg = 255;

        if let Ok(save) = game_impl.save_from_wram(wram) {
            let assets = game_impl.load_rom_assets(&rom, wram, &rom::Overrides::default()).ok();
            
            if let Some(assets_ref) = assets.as_ref() {
                p_navi = extract_navi_cust_ids(&*save, assets_ref.as_ref());
            }
            
            let chips = extract_chips(&*save);
            for c in chips { p_ids.push(c.id as u16); p_codes.push(c.code as u16); }
            
            let tags = extract_tags(&*save);
            for t in tags { p_tags.push(t as u16); }
            
            p_reg = extract_reg(&*save) as u16;
        }

        // Remote Player (Enemy)
        let wram_remote = replay.remote_state.wram();
        let mut e_navi = Vec::new();
        let mut e_ids = Vec::new();
        let mut e_codes = Vec::new();
        let mut e_tags = Vec::new();
        let mut e_reg = 255;

        if let Ok(save_remote) = game_impl.save_from_wram(wram_remote) {
            let assets = game_impl.load_rom_assets(&rom, wram_remote, &rom::Overrides::default()).ok();
            if let Some(assets_ref) = assets.as_ref() {
                e_navi = extract_navi_cust_ids(&*save_remote, assets_ref.as_ref());
            }
            
            let chips = extract_chips(&*save_remote);
            for c in chips { e_ids.push(c.id as u16); e_codes.push(c.code as u16); }
            
            let tags = extract_tags(&*save_remote);
            for t in tags { e_tags.push(t as u16); }
            
            e_reg = extract_reg(&*save_remote) as u16;
        }

        let static_data = StaticReplayData {
            player_navi_cust: p_navi,
            enemy_navi_cust: e_navi,
            player_folder_ids: p_ids,
            player_folder_codes: p_codes,
            enemy_folder_ids: e_ids,
            enemy_folder_codes: e_codes,
            player_reg_chip: p_reg,
            enemy_reg_chip: e_reg,
            player_tags: p_tags,
            enemy_tags: e_tags,
        };

        let f = std::fs::File::create("static_data.json").expect("Failed to create static_data.json");
        serde_json::to_writer(f, &static_data).expect("Failed to write static_data.json");
        println!("✅ Static Data Saved to static_data.json");
    } else {
        println!("⚠️ Warning: Could not resolve Game Implementation. Static data skipped.");
    }
    // -----------------------------------------

    let hooks = tango_pvp::hooks::hooks_for_gamedb_entry(detected_game)
        .ok_or(anyhow::anyhow!("No hooks found."))?;

    let settings = tango_pvp::replay::export::Settings {
        ffmpeg: None,
        ffmpeg_audio_flags: "-c:a aac -ar 48000 -b:a 384k -ac 2".to_string(),
        ffmpeg_video_flags: "-c:v libx264 -vf scale=iw*2:ih*2:flags=neighbor,format=yuv420p -force_key_frames expr:gte(t,n_forced/2) -crf 18 -bf 2".to_string(),
        ffmpeg_mux_flags: "-movflags +faststart -strict -2".to_string(),
        disable_bgm: false,
    };

    let bar = indicatif::ProgressBar::new(0);
    bar.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());
    
    let cb = move |current, total| {
        bar.set_length(total as u64);
        bar.set_position(current as u64);
    };

    tango_pvp::replay::export::export(
        &rom,
        hooks,
        &[replay],
        &output_path,
        &settings,
        cb,
    ).await?;

    Ok(())
}

// --- CHILD MAIN ---

fn child_main(mut config: config::Config, args: EnvArgs) -> Result<(), anyhow::Error> {
    let init_link_code = args.init_link_code;
    let rom_path = args.rom;
    let save_path = args.save;
    let port = args.port;

    if let Some(rpath) = args.replay_path {
        global::set_replay_path(rpath);
    }

    let rt = Runtime::new()?;

    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<InputCommand>(); 
    let (output_tx, _output_rx) = mpsc::unbounded_channel::<OutputMessage>(); // _output_rx to silence warning
    let output_tx = Arc::new(Mutex::new(Some(output_tx))); 

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

    let event_loop = winit::event_loop::EventLoopBuilder::<UserEvent>::with_user_event().build().unwrap();

    let icon = image::load_from_memory(include_bytes!("icon.png"))?;
    let icon_width = icon.width();
    let icon_height = icon.height();
    let window_title = format!("{}", args.port);

    let window_builder = winit::window::WindowBuilder::new()
        .with_title(&window_title) 
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
            winit::event::Event::WindowEvent { event: window_event, .. } => {
                match window_event {
                    winit::event::WindowEvent::RedrawRequested if !cfg!(windows) => redraw(),
                    winit::event::WindowEvent::KeyboardInput {
                        event: winit::event::KeyEvent {
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
                            winit::event::WindowEvent::Occluded(false) => {
                                next_config.full_screen = gfx_backend.window().fullscreen().is_some();
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
            winit::event::Event::AboutToWait => {
                if cfg!(windows) { redraw(); }

                while let Ok(cmd) = input_rx.try_recv() {
                    input_state.clear_keys();
                    match cmd.command_type.as_str() {
                        "key_press" => {
                            for (i, bit) in cmd.key.chars().rev().enumerate() {
                                if bit == '1' {
                                    if let Some(key) = map_bit_to_key(i) {
                                        handle_input_event(
                                            &mut input_state,
                                            &mut state,
                                            key,
                                            winit::event::ElementState::Pressed,
                                            &mut next_config,
                                        );
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        if let Some(session) = state.shared.session.lock().as_mut() {
            session.set_joyflags(next_config.input_mapping.to_mgba_keys(&input_state));
            session.set_master_volume(next_config.volume);
        }

        if let Some(player_won) = get_winner() {
            if let Some(ref output_tx) = *output_tx.lock() {
                let message = OutputMessage {
                    event: "winner".to_string(),
                    details: format!("{}", player_won),
                };
                let _ = output_tx.send(message);
            }
            std::process::exit(0);
        }

        next_config.window_size = gfx_backend.window().inner_size().to_logical(gfx_backend.window().scale_factor());

        if next_config != old_config {
            last_config_dirty_time = Some(std::time::Instant::now());
            *config.write() = next_config.clone();
        }

        if last_config_dirty_time.map(|t| (std::time::Instant::now() - t) > std::time::Duration::from_secs(1)).unwrap_or(false) {
            let _ = next_config.save();
            last_config_dirty_time = None;
        }

        gfx_backend.set_ui_scale(next_config.ui_scale_percent as f32 / 100.0);
        patch_autoupdater.set_enabled(next_config.enable_patch_autoupdate);
        updater.set_enabled(next_config.enable_updater);
    })?;

    Ok(())
}

fn map_bit_to_key(bit: usize) -> Option<Key> {
    match bit {
        8 => Some(Key::A),      
        7 => Some(Key::Down),   
        6 => Some(Key::Up),     
        5 => Some(Key::Left),   
        4 => Some(Key::Right),  
        9 => Some(Key::S),      
        1 => Some(Key::X),      
        0 => Some(Key::Z),      
        3 => Some(Key::Return), 
        _ => None,
    }
}

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

#[derive(Deserialize, Serialize, Debug, Clone)]
struct InputCommand {
    #[serde(rename = "type")]
    command_type: String,
    key: String,
}

#[derive(Serialize, Debug)]
struct OutputMessage {
    event: String,
    details: String,
}

#[derive(Serialize, Debug)]
struct ScreenImageDetails {
    image: String,
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
    grid_state : Vec<u16>,
    grid_owner_state : Vec<u16>,
    player_grid_position: Vec<u16>,
    enemy_grid_position: Vec<u16>,
    is_offerer: u16,
    cust_gage: u16,
    own_navi_cust: Vec<u16>,
    enemy_navi_cust: Vec<u16>,
}

async fn setup_tcp_listener(
    port: u16,
    tx: mpsc::UnboundedSender<InputCommand>,
    output_tx: Arc<Mutex<Option<mpsc::UnboundedSender<OutputMessage>>>>,
) -> Result<(), anyhow::Error> {
    let listener = TcpListener::bind(("127.0.0.1", port)).await?;
    println!("Listening for input events on port {}", port);

    loop {
        match listener.accept().await {
            Ok((socket, addr)) => {
                println!("Accepted connection from {} on port {}", addr, port);
                if let Err(e) = socket.set_nodelay(true) {
                    println!("Warning: Failed to set TCP_NODELAY: {}", e);
                }
                tokio::spawn(handle_tcp_client(socket, tx.clone(), output_tx.clone()));
            }
            Err(e) => {
                println!("Failed to accept connection: {}", e);
            }
        }
    }
}

async fn handle_tcp_client(
    mut socket: tokio::net::TcpStream,
    tx: mpsc::UnboundedSender<InputCommand>,
    output_tx: Arc<Mutex<Option<mpsc::UnboundedSender<OutputMessage>>>>,
) {
    let (msg_tx, mut msg_rx) = mpsc::unbounded_channel::<OutputMessage>();
    *output_tx.lock() = Some(msg_tx);

    let mut rgb_bytes: Vec<u8> = Vec::with_capacity(240 * 160 * 3);
    let mut img_data_buffer: Vec<u8> = Vec::with_capacity(1024 * 50);

    let mut buf = vec![0; 8192];
    loop {
        tokio::select! {
            n = socket.read(&mut buf) => {
                match n {
                    Ok(0) => break,
                    Ok(n) => {
                        let data = &buf[..n];
                        if let Ok(command_str) = std::str::from_utf8(data) {
                            for line in command_str.lines() {
                                if let Ok(cmd) = serde_json::from_str::<InputCommand>(line) {
                                    match cmd.command_type.as_str() {
                                        "request_screen" => {
                                            if let Some(image) = get_screen_image() {
                                                rgb_bytes.clear();
                                                for pixel in &image.pixels {
                                                    rgb_bytes.push(pixel.r());
                                                    rgb_bytes.push(pixel.g());
                                                    rgb_bytes.push(pixel.b());
                                                }

                                                img_data_buffer.clear();

                                                // [FIX START] Switch from JpegEncoder to PngEncoder
                                                // PNG is lossless. It will be slightly slower to encode than JPEG,
                                                // but for 240x160 resolution, it is negligible.
                                                let encoder = PngEncoder::new(&mut img_data_buffer);
                                                
                                                // Encode the image
                                                if let Err(e) = encoder.write_image(
                                                    &rgb_bytes,
                                                    image.size[0] as u32,
                                                    image.size[1] as u32,
                                                    image::ExtendedColorType::Rgb8, 
                                                ) {
                                                    println!("Failed to encode image: {}", e);
                                                    continue;
                                                }
                                                // [FIX END]

                                                let encoded_image = encode(&img_data_buffer);

                                                let player_health = get_player_health();
                                                let enemy_health = get_enemy_health();
                                                
                                                let screen_details = ScreenImageDetails {
                                                    image: encoded_image,
                                                    player_health,
                                                    enemy_health,
                                                    player_position: get_player_position(),
                                                    enemy_position: get_enemy_position(),
                                                    inside_window: get_is_player_inside_window().unwrap_or(false),
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
                                                    grid_state: get_all_grid_states(),
                                                    grid_owner_state: get_all_grid_owner_states(),
                                                    player_grid_position: get_player_grid_position(),
                                                    enemy_grid_position: get_enemy_grid_position(),
                                                    is_offerer: get_is_offerer(),
                                                    cust_gage: get_cust_gage(),
                                                    own_navi_cust: get_player_navi_cust_parts().unwrap_or_default().into_iter().map(|x| x as u16).collect(),
                                                    enemy_navi_cust: get_enemy_navi_cust_parts().unwrap_or_default().into_iter().map(|x| x as u16).collect(),
                                                };

                                                clear_local_input();
                                                clear_rewards();
                                                clear_punishments();

                                                let details_json = serde_json::to_string(&screen_details)
                                                    .expect("Failed to serialize screen details");

                                                let response = OutputMessage {
                                                    event: "screen_image".to_string(),
                                                    details: details_json,
                                                };

                                                if player_health == 0 && enemy_health != 0
                                                || enemy_health == 0 && player_health != 0 {
                                                    println!("Game Over");
                                                    std::process::exit(0);
                                                }

                                                if let Err(e) = send_message_to_python(&mut socket, &response).await {
                                                    println!("Failed to send screen image: {}", e);
                                                }
                                            }
                                        }
                                        _ => {
                                            if tx.send(cmd.clone()).is_err() { }
                                            let response = OutputMessage {
                                                event: "command_received".to_string(),
                                                details: format!("Processed command: {:?}", cmd),
                                            };
                                            let _ = send_message_to_python(&mut socket, &response).await;
                                        }
                                    }
                                } else {
                                    println!("Failed to parse input command: {}", line);
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
            Some(message) = msg_rx.recv() => {
                if let Err(e) = send_message_to_python(&mut socket, &message).await {
                    println!("Failed to send message to Python: {}", e);
                }
            }
            else => break,
        }
    }
}

async fn send_message_to_python(
    socket: &mut tokio::net::TcpStream,
    message: &OutputMessage,
) -> Result<(), Box<dyn std::error::Error>> {
    let message_json = serde_json::to_string(message)?;
    socket.write_all(message_json.as_bytes()).await?;
    socket.write_all(b"\n").await?;
    Ok(())
}