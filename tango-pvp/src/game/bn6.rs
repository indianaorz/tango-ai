// tango-pvp/src/game/bn6.rs

mod munger;
mod offsets;

pub struct Hooks {
    offsets: &'static offsets::Offsets,
}

impl Hooks {
    fn munger(&self) -> munger::Munger {
        munger::Munger { offsets: self.offsets }
    }
}

pub static BR6E_00: Hooks = Hooks {
    offsets: &offsets::MEGAMAN6_FXXBR6E_00,
};
pub static BR5E_00: Hooks = Hooks {
    offsets: &offsets::MEGAMAN6_GXXBR5E_00,
};
pub static BR6J_00: Hooks = Hooks {
    offsets: &offsets::ROCKEXE6_RXXBR6J_00,
};
pub static BR5J_00: Hooks = Hooks {
    offsets: &offsets::ROCKEXE6_GXXBR5J_00,
};

fn generate_rng1_state(rng: &mut impl rand::Rng) -> u32 {
    (0..rng.gen_range(0..0x100000)).fold(0, |acc, _| step_rng(acc))
}

fn generate_rng2_state(rng: &mut impl rand::Rng) -> u32 {
    (0..rng.gen_range(0..0x100000)).fold(0xa338244f, |acc, _| step_rng(acc))
}

fn random_battle_settings_and_background(rng: &mut impl rand::Rng, match_type: u8) -> u16 {
    const BATTLE_BACKGROUNDS: &[u16] = &[
        0x00, 0x01, 0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11,
        0x11, 0x13, 0x13,
    ];

    let lo = match match_type {
        0 => rng.gen_range(0..0x44u16),
        1 => rng.gen_range(0..0x60u16),
        2 => rng.gen_range(0..0x44u16) + 0x60u16,
        _ => 0u16,
    };

    let hi = BATTLE_BACKGROUNDS[rng.gen_range(0..BATTLE_BACKGROUNDS.len())];

    hi << 0x8 | lo
}

fn step_rng(seed: u32) -> u32 {
    let seed = std::num::Wrapping(seed);
    ((seed << 1) + (seed >> 0x1f) + std::num::Wrapping(1)).0 ^ 0x873ca9e5
}

impl crate::hooks::Hooks for Hooks {
    fn common_traps(&self) -> Vec<(u32, Box<dyn Fn(mgba::core::CoreMutRef)>)> {
        vec![
            (self.offsets.rom.start_screen_jump_table_entry, {
                let munger = self.munger();
                Box::new(move |core| {
                    munger.skip_logo(core);
                })
            }),
            (self.offsets.rom.start_screen_sram_unmask_ret, {
                let munger = self.munger();
                Box::new(move |core| {
                    munger.continue_from_title_menu(core);
                })
            }),
            (self.offsets.rom.game_load_ret, {
                let munger = self.munger();
                Box::new(move |core| {
                    munger.open_comm_menu_from_overworld(core);
                })
            }),
        ]
    }

/// Per-frame telemetry snapshot for replay export.
fn capture_frame_telemetry(&self, mut core: mgba::core::CoreMutRef) -> Option<crate::telemetry::FrameTelemetry> {
        // --- 1. MEMORY ADDRESSES ---
        // Obj0 (Default P1 / Host / Left)
        const HP_OBJ0_ADDR: u32 = 0x0203A9D4;
        const X_OBJ0_ADDR: u32 = 0x0203AA4C; 
        const Y_OBJ0_ADDR: u32 = 0x0203A9C4;
        const CHARGE_OBJ0_ADDR: u32 = 0x0203409D;
        const CHIP_OBJ0_ADDR: u32 = 0x0203A9DA; 
        const EMO_GAME_OBJ0_ADDR: u32 = 0x0203CE90;

        // Obj1 (Default P2 / Client / Right)
        const HP_OBJ1_ADDR: u32 = 0x0203AAAC;
        const X_OBJ1_ADDR: u32 = 0x0203AB24;
        const Y_OBJ1_ADDR: u32 = 0x0203AA9C;
        const CHARGE_OBJ1_ADDR: u32 = 0x0203419D;
        const CHIP_OBJ1_ADDR: u32 = 0x0203AAB2;
        const EMO_GAME_OBJ1_ADDR: u32 = 0x0203CE2C;

        // Local Console State (Always "My" view)
        const EMO_WIN_LOCAL_ADDR: u32 = 0x020352CC; 
        const CUST_GAUGE_ADDR: u32 = 0x020352A1;
        const WINDOW_ADDR: u32 = 0x02035288; 
        const BEAST_SELECTABLE_ADDR: u32 = 0x0203664B;
        const CROSS_WINDOW_ADDR: u32 = 0x020364C2;
        const CHIP_SEL_COUNT_ADDR: u32 = 0x020364C8;
        const CHIP_VISIBLE_COUNT_ADDR: u32 = 0x020047D6;
        const MENU_INDEX_ADDR: u32 = 0x020364C7;
        const CROSS_INDEX_ADDR: u32 = 0x020364DB;
        const SELECTED_INDICES_BASE: u32 = 0x02036508; 

        // Grids
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

        // --- 2. READ RAW DATA (No Swapping Logic) ---
        // We strictly map "player_*" to Obj0 and "enemy_*" to Obj1.
        // If the recording is actually from P2's perspective, the Python script will detect this and swap them.
        
        let p_hp = core.raw_read_16(HP_OBJ0_ADDR, -1);
        let e_hp = core.raw_read_16(HP_OBJ1_ADDR, -1);
        
        // Sanity check: If both are 0, we are likely not in a match yet.
        if p_hp == 0 && e_hp == 0 { return None; }

        let p_x = core.raw_read_16(X_OBJ0_ADDR, -1);
        let p_y = core.raw_read_16(Y_OBJ0_ADDR, -1);
        let e_x = core.raw_read_16(X_OBJ1_ADDR, -1);
        let e_y = core.raw_read_16(Y_OBJ1_ADDR, -1);

        let p_chg = core.raw_read_8(CHARGE_OBJ0_ADDR, -1) as u16;
        let e_chg = core.raw_read_8(CHARGE_OBJ1_ADDR, -1) as u16;
        let p_chip = core.raw_read_16(CHIP_OBJ0_ADDR, -1);
        let e_chip = core.raw_read_16(CHIP_OBJ1_ADDR, -1);
        
        let  e_game_emo= core.raw_read_8(EMO_GAME_OBJ0_ADDR, -1) as u16;
        let p_game_emo = core.raw_read_8(EMO_GAME_OBJ1_ADDR, -1) as u16;

        // --- 3. READ LOCAL DATA ---
        let cust_gauge = core.raw_read_8(CUST_GAUGE_ADDR, -1) as u16;
        let inside_window = core.raw_read_8(WINDOW_ADDR, -1) == 255;
        let p_win_emo = core.raw_read_8(EMO_WIN_LOCAL_ADDR, -1) as u16; 
        let beast = core.raw_read_8(BEAST_SELECTABLE_ADDR, -1) as u16;
        let cross = core.raw_read_8(CROSS_WINDOW_ADDR, -1) as u16;
        let sel_count = core.raw_read_8(CHIP_SEL_COUNT_ADDR, -1) as u16;
        let vis_count = core.raw_read_8(CHIP_VISIBLE_COUNT_ADDR, -1) as u16;
        let menu_idx = core.raw_read_8(MENU_INDEX_ADDR, -1) as u16;
        let cross_idx = core.raw_read_8(CROSS_INDEX_ADDR, -1) as u16;

        let mut selected_indices = Vec::with_capacity(5);
        for i in 0..5 {
            selected_indices.push(core.raw_read_8(SELECTED_INDICES_BASE + i, -1) as u16);
        }

        let mut grid_state = Vec::with_capacity(18);
        for addr in &grid_addresses { grid_state.push(core.raw_read_8(*addr, -1) as u16); }
        let mut grid_owner_state = Vec::with_capacity(18);
        for addr in &grid_owner_addresses { grid_owner_state.push(core.raw_read_8(*addr, -1) as u16); }

        let mut chip_slots = Vec::new();
        let mut chip_codes = Vec::new();
        for i in 0..16 {
             let address = 0x0203CDB0 + i as u32;
             let val = core.raw_read_8(address, -1) as u16;
             if i % 2 == 0 { chip_slots.push(val); } 
             else { chip_codes.push(val); }
        }

        // --- 4. CONSTRUCT TELEMETRY ---
        Some(crate::telemetry::FrameTelemetry::V1(crate::telemetry::FrameTelemetryV1 {
            player_health: Some(p_hp),
            enemy_health: Some(e_hp),
            player_pos: Some((p_x, p_y)),
            enemy_pos: Some((e_x, e_y)),
            player_charge: Some(p_chg),
            enemy_charge: Some(e_chg),
            player_chip: Some(p_chip),
            enemy_chip: Some(e_chip),
            
            // Emotions
            player_emotion: Some(p_win_emo), // Window Face (Always Local)
            enemy_emotion: Some(0),          // Window Face (Not visible for enemy)
            
            player_game_emotion: Some(p_game_emo), // Grid State (Obj0)
            enemy_game_emotion: Some(e_game_emo),  // Grid State (Obj1)

            cust_gauge: Some(cust_gauge),
            inside_window: Some(inside_window),
            beast_mode: Some(beast),
            cross_window: Some(cross),
            chip_select_count: Some(sel_count),
            chip_visible_count: Some(vis_count),
            selected_menu_index: Some(menu_idx),
            selected_cross_index: Some(cross_idx),

            grid_state: Some(grid_state),
            grid_owner_state: Some(grid_owner_state),
            
            chip_slots: Some(chip_slots),
            chip_codes: Some(chip_codes),
            selected_chip_indices: Some(selected_indices),

            // Static Placeholders
            player_folder: None, enemy_folder: None,
            player_code_folder: None, enemy_code_folder: None,
            player_tag_chips: None, enemy_tag_chips: None,
            player_reg_chip: None, enemy_reg_chip: None,
            player_navi_cust: None, enemy_navi_cust: None,
        }))
    }
    fn prepare_for_fastforward(&self, mut core: mgba::core::CoreMutRef) {
        core.gba_mut()
            .cpu_mut()
            .set_thumb_pc(self.offsets.rom.main_read_joyflags);
    }

fn primary_traps(
        &self,
        joyflags: std::sync::Arc<std::sync::atomic::AtomicU32>,
        match_: std::sync::Arc<tokio::sync::Mutex<Option<std::sync::Arc<crate::battle::Match>>>>,
        completion_token: crate::hooks::CompletionToken,
    ) -> Vec<(u32, Box<dyn Fn(mgba::core::CoreMutRef)>)> {
        vec![
            (self.offsets.rom.comm_menu_init_ret, {
                let match_ = match_.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    munger.start_battle_from_comm_menu(core, match_.match_type().0);
                })
            }),
            (self.offsets.rom.round_end_set_win, {
                let match_ = match_.clone();
                Box::new(move |_| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Win);
                })
            }),
            (self.offsets.rom.round_end_set_loss, {
                let match_ = match_.clone();
                Box::new(move |_| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Loss);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_win, {
                let match_ = match_.clone();
                Box::new(move |_| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Win);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_loss, {
                let match_ = match_.clone();
                Box::new(move |_| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Loss);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_draw, {
                let match_ = match_.clone();
                Box::new(move |_| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();
                    let result = {
                        let round = round_state.round.as_ref().expect("round");
                        round.on_draw_outcome()
                    };
                    round_state.set_last_outcome(result);
                })
            }),
            (self.offsets.rom.round_set_ending, {
                let match_ = match_.clone();
                Box::new(move |_| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();
                    round_state.end_round().expect("end round");
                    match_.advance_shadow_until_round_end().expect("advance shadow");
                })
            }),
            (self.offsets.rom.round_start_ret, {
                let match_ = match_.clone();
                Box::new(move |_core| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };
                    crate::sync::block_on(match_.start_round()).expect("start round");
                })
            }),
            (self.offsets.rom.battle_is_p2_tst, {
                let match_ = match_.clone();
                Box::new(move |mut core| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let round_state = match_.lock_round_state();
                    let round = round_state.round.as_ref().expect("round");

                    core.gba_mut().cpu_mut().set_gpr(0, round.local_player_index() as i32);
                })
            }),
            (self.offsets.rom.link_is_p2_ret, {
                let match_ = match_.clone();
                Box::new(move |mut core| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let round_state = match_.lock_round_state();
                    let round = round_state.round.as_ref().expect("round");

                    core.gba_mut().cpu_mut().set_gpr(0, round.local_player_index() as i32);
                })
            }),
            (
                self.offsets.rom.handle_sio_entry,
                Box::new(move |core| {
                    log::error!(
                        "unhandled call to handleSIO at 0x{:0x}: uh oh!",
                        core.as_ref().gba().cpu().gpr(14) - 2
                    );
                }),
            ),
            (self.offsets.rom.comm_menu_init_battle_entry, {
                let match_ = match_.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    // [FIX] Lock and unwrap the match object first
                    let match_lock = match_.blocking_lock();
                    let match_inner = match &*match_lock {
                        Some(m) => m,
                        None => return,
                    };

                    let mut rng = match_inner.lock_rng();
                    munger.set_link_battle_settings_and_background(
                        core,
                        random_battle_settings_and_background(&mut *rng, match_inner.match_type().0),
                    );
                })
            }),
            (
                self.offsets.rom.comm_menu_end_battle_entry,
                Box::new(move |_core| {
                    completion_token.complete();
                }),
            ),
            (
                self.offsets
                    .rom
                    .comm_menu_in_battle_call_comm_menu_handle_link_cable_input,
                {
                    let match_ = match_.clone();
                    let munger = self.munger();
                    Box::new(move |mut core| {
                        let pc = core.as_ref().gba().cpu().thumb_pc();
                        core.gba_mut().cpu_mut().set_thumb_pc(pc + 6);
                        munger.set_copy_data_input_state(core, if match_.blocking_lock().is_some() { 2 } else { 4 });
                    })
                },
            ),
            (self.offsets.rom.main_read_joyflags, {
                let match_ = match_.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let match_ = match_.blocking_lock();
                    let match_ = match &*match_ {
                        Some(match_) => match_,
                        _ => {
                            return;
                        }
                    };

                    let mut round_state = match_.lock_round_state();

                    let round = match round_state.round.as_mut() {
                        Some(round) => round,
                        None => {
                            return;
                        }
                    };

                    if !round.has_committed_state() {
                        let mut rng = match_.lock_rng();

                        // rng1 is the local rng, it should not be synced.
                        // However, we should make sure it's reproducible from the shared RNG state so we generate it like this.
                        let offerer_rng1_state = generate_rng1_state(&mut *rng);
                        let answerer_rng1_state = generate_rng1_state(&mut *rng);
                        munger.set_rng1_state(
                            core,
                            if match_.is_offerer() {
                                offerer_rng1_state
                            } else {
                                answerer_rng1_state
                            },
                        );

                        // rng2 is the shared rng, it must be synced.
                        let rng2_state = generate_rng2_state(&mut *rng);
                        munger.set_rng2_state(core, rng2_state);

                        // HACK: The battle jump table goes directly from deinit to init, so we actually end up initializing on tick 1 after round 1. We just override it here.
                        munger.set_current_tick(core, 0);

                        round.set_first_committed_state(
                            core.save_state().expect("save state"),
                            match_
                                .advance_shadow_until_first_committed_state()
                                .expect("shadow save state"),
                            &munger.tx_packet(core),
                        );

                        log::info!(
                            "primary rng1 state: {:08x}, rng2 state: {:08x}",
                            munger.rng1_state(core),
                            munger.rng2_state(core),
                        );
                        log::info!("battle state committed on {}", round.current_tick());
                    }

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != round.current_tick() {
                        panic!(
                            "read joyflags: round tick = {} but game tick = {}",
                            round.current_tick(),
                            game_current_tick
                        );
                    }

                    if let Err(e) = crate::sync::block_on(round.add_local_input_and_fastforward(
                        core,
                        joyflags.load(std::sync::atomic::Ordering::Relaxed) as u16,
                    )) {
                        log::error!("failed to add local input: {}", e);
                        match_.cancel();
                    }
                })
            }),
            {
                let match_ = match_.clone();
                let munger = self.munger();
                (
                    self.offsets.rom.round_post_increment_tick,
                    Box::new(move |core| {
                        let match_ = match_.blocking_lock();
                        let match_ = match &*match_ {
                            Some(match_) => match_,
                            _ => {
                                return;
                            }
                        };

                        let mut round_state = match_.lock_round_state();

                        let round = match round_state.round.as_mut() {
                            Some(round) => round,
                            None => {
                                return;
                            }
                        };

                        if !round.has_committed_state() {
                            return;
                        }

                        round.increment_current_tick();
                        let game_current_tick = munger.current_tick(core);
                        if game_current_tick != round.current_tick() {
                            panic!(
                                "post increment tick: round tick = {} but game tick = {}",
                                round.current_tick(),
                                game_current_tick
                            );
                        }
                    }),
                )
            },
        ]
    }

    fn shadow_traps(&self, shadow_state: crate::shadow::State) -> Vec<(u32, Box<dyn Fn(mgba::core::CoreMutRef)>)> {
        vec![
            (self.offsets.rom.comm_menu_init_ret, {
                let munger = self.munger();
                let shadow_state = shadow_state.clone();
                Box::new(move |core| {
                    munger.start_battle_from_comm_menu(core, shadow_state.match_type().0);
                })
            }),
            (self.offsets.rom.round_end_set_win, {
                let shadow_state = shadow_state.clone();
                Box::new(move |_| {
                    let mut round_state = shadow_state.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Loss);
                })
            }),
            (self.offsets.rom.round_end_set_loss, {
                let shadow_state = shadow_state.clone();
                Box::new(move |_| {
                    let mut round_state = shadow_state.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Win);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_win, {
                let shadow_state = shadow_state.clone();
                Box::new(move |_| {
                    let mut round_state = shadow_state.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Loss);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_loss, {
                let shadow_state = shadow_state.clone();
                Box::new(move |_| {
                    let mut round_state = shadow_state.lock_round_state();
                    round_state.set_last_outcome(crate::battle::BattleOutcome::Win);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_draw, {
                let shadow_state = shadow_state.clone();
                Box::new(move |_| {
                    let mut round_state = shadow_state.lock_round_state();
                    let result = {
                        let round = round_state.round.as_mut().expect("round");
                        round.on_draw_outcome()
                    };
                    round_state.set_last_outcome(result);
                })
            }),
            (self.offsets.rom.round_start_ret, {
                let shadow_state = shadow_state.clone();
                Box::new(move |_| {
                    shadow_state.start_round();
                })
            }),
            (self.offsets.rom.round_end_entry, {
                let shadow_state = shadow_state.clone();
                Box::new(move |core| {
                    shadow_state.end_round();
                    shadow_state.set_applied_state(core.save_state().expect("save state"), 0);
                })
            }),
            (self.offsets.rom.battle_is_p2_tst, {
                let shadow_state = shadow_state.clone();
                Box::new(move |mut core| {
                    let mut round_state = shadow_state.lock_round_state();
                    let round = round_state.round.as_mut().expect("round");

                    core.gba_mut().cpu_mut().set_gpr(0, round.remote_player_index() as i32);
                })
            }),
            (self.offsets.rom.link_is_p2_ret, {
                let shadow_state = shadow_state.clone();
                Box::new(move |mut core| {
                    let mut round_state = shadow_state.lock_round_state();
                    let round = round_state.round.as_mut().expect("round");

                    core.gba_mut().cpu_mut().set_gpr(0, round.remote_player_index() as i32);
                })
            }),
            (
                self.offsets.rom.handle_sio_entry,
                Box::new(move |core| {
                    log::error!(
                        "unhandled call to handleSIO at 0x{:0x}: uh oh!",
                        core.as_ref().gba().cpu().gpr(14) - 2
                    );
                }),
            ),
            (self.offsets.rom.comm_menu_init_battle_entry, {
                let shadow_state = shadow_state.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let mut rng = shadow_state.lock_rng();
                    munger.set_link_battle_settings_and_background(
                        core,
                        random_battle_settings_and_background(&mut *rng, shadow_state.match_type().0),
                    );
                })
            }),
            (
                self.offsets
                    .rom
                    .comm_menu_in_battle_call_comm_menu_handle_link_cable_input,
                {
                    let munger = self.munger();
                    Box::new(move |mut core| {
                        let pc = core.as_ref().gba().cpu().thumb_pc();
                        core.gba_mut().cpu_mut().set_thumb_pc(pc + 6);
                        munger.set_copy_data_input_state(core, 2);
                    })
                },
            ),
            (self.offsets.rom.main_read_joyflags, {
                let shadow_state = shadow_state.clone();
                let munger = self.munger();
                Box::new(move |mut core| {
                    let mut round_state = shadow_state.lock_round_state();
                    let round = match round_state.round.as_mut() {
                        Some(round) => round,
                        None => {
                            return;
                        }
                    };

                    if !round.has_first_committed_state() {
                        let mut rng = shadow_state.lock_rng();

                        // rng1 is the local rng, it should not be synced.
                        // However, we should make sure it's reproducible from the shared RNG state so we generate it like this.
                        let offerer_rng1_state = generate_rng1_state(&mut *rng);
                        let answerer_rng1_state = generate_rng1_state(&mut *rng);
                        munger.set_rng1_state(
                            core,
                            if shadow_state.is_offerer() {
                                answerer_rng1_state
                            } else {
                                offerer_rng1_state
                            },
                        );

                        // rng2 is the shared rng, it must be synced.
                        let rng2_state = generate_rng2_state(&mut *rng);
                        munger.set_rng2_state(core, rng2_state);

                        // HACK: The battle jump table goes directly from deinit to init, so we actually end up initializing on tick 1 after round 1. We just override it here.
                        munger.set_current_tick(core, 0);

                        round
                            .set_first_committed_state(core.save_state().expect("save state"), &munger.tx_packet(core));
                        log::info!(
                            "shadow rng1 state: {:08x}, rng2 state: {:08x}",
                            munger.rng1_state(core),
                            munger.rng2_state(core)
                        );
                        log::info!("shadow state committed on {}", round.current_tick());
                        return;
                    }

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != round.current_tick() {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "read joyflags: round tick = {} but game tick = {}",
                            round.current_tick(),
                            game_current_tick
                        ));
                    }

                    if let Some(ip) = round.peek_shadow_input().clone() {
                        if ip.local.local_tick != ip.remote.local_tick {
                            shadow_state.set_anyhow_error(anyhow::anyhow!(
                                "read joyflags: local tick != remote tick (in battle tick = {}): {} != {}",
                                round.current_tick(),
                                ip.local.local_tick,
                                ip.remote.local_tick
                            ));
                            return;
                        }

                        if ip.local.local_tick != round.current_tick() {
                            shadow_state.set_anyhow_error(anyhow::anyhow!(
                                "read joyflags: input tick != in battle tick: {} != {}",
                                ip.local.local_tick,
                                round.current_tick(),
                            ));
                            return;
                        }

                        core.gba_mut()
                            .cpu_mut()
                            .set_gpr(4, (ip.remote.joyflags | 0xfc00) as i32);
                    }

                    if round.take_input_injected() {
                        shadow_state.set_applied_state(core.save_state().expect("save state"), round.current_tick());
                    }
                })
            }),
            (self.offsets.rom.copy_input_data_entry, {
                let shadow_state = shadow_state.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let mut round_state = shadow_state.lock_round_state();
                    let round = round_state.round.as_mut().expect("round");

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != round.current_tick() {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: round tick = {} but game tick = {}",
                            round.current_tick(),
                            game_current_tick
                        ));
                    }

                    let ip = if let Some(ip) = round.take_shadow_input() {
                        ip
                    } else {
                        return;
                    };

                    // HACK: This is required if the emulator advances beyond read joyflags and runs this function again, but is missing input data.
                    // We permit this for one tick only, but really we should just not be able to get into this situation in the first place.
                    if ip.local.local_tick + 1 == round.current_tick() {
                        return;
                    }

                    if ip.local.local_tick != ip.remote.local_tick {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: local tick != remote tick (in battle tick = {}): {} != {}",
                            round.current_tick(),
                            ip.local.local_tick,
                            ip.remote.local_tick
                        ));
                        return;
                    }

                    if ip.local.local_tick != round.current_tick() {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: input tick != in battle tick: {} != {}",
                            ip.local.local_tick,
                            round.current_tick(),
                        ));
                        return;
                    }

                    let remote_packet = round.peek_remote_packet().unwrap();
                    if remote_packet.tick != round.current_tick() {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: local packet tick != in battle tick: {} != {}",
                            remote_packet.tick,
                            round.current_tick(),
                        ));
                        return;
                    }

                    munger.set_rx_packet(
                        core,
                        round.local_player_index() as u32,
                        &ip.local.packet.try_into().unwrap(),
                    );
                    munger.set_rx_packet(
                        core,
                        round.remote_player_index() as u32,
                        &remote_packet.packet.clone().try_into().unwrap(),
                    );
                })
            }),
            (self.offsets.rom.copy_input_data_ret, {
                let shadow_state = shadow_state.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let mut round_state = shadow_state.lock_round_state();
                    let round = round_state.round.as_mut().expect("round");

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != round.current_tick() {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: round tick = {} but game tick = {}",
                            round.current_tick(),
                            game_current_tick
                        ));
                    }

                    round.set_remote_packet(round.current_tick() + 1, munger.tx_packet(core).to_vec());
                    round.set_input_injected();
                })
            }),
            (self.offsets.rom.round_post_increment_tick, {
                let shadow_state = shadow_state.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let mut round_state = shadow_state.lock_round_state();
                    let round = round_state.round.as_mut().expect("round");
                    if !round.has_first_committed_state() {
                        return;
                    }
                    round.increment_current_tick();

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != round.current_tick() {
                        shadow_state.set_anyhow_error(anyhow::anyhow!(
                            "post increment tick: round tick = {} but game tick = {}",
                            round.current_tick(),
                            game_current_tick
                        ));
                    }
                })
            }),
        ]
    }

    fn stepper_traps(&self, stepper_state: crate::stepper::State) -> Vec<(u32, Box<dyn Fn(mgba::core::CoreMutRef)>)> {
        vec![
            (self.offsets.rom.battle_start_play_music_call, {
                let stepper_state = stepper_state.clone();
                Box::new(move |mut core| {
                    let stepper_state = stepper_state.lock_inner();
                    if !stepper_state.disable_bgm() {
                        return;
                    }
                    let pc = core.as_ref().gba().cpu().thumb_pc();
                    core.gba_mut().cpu_mut().set_thumb_pc(pc + 4);
                })
            }),
            (self.offsets.rom.battle_is_p2_tst, {
                let stepper_state = stepper_state.clone();
                Box::new(move |mut core| {
                    let stepper_state = stepper_state.lock_inner();
                    core.gba_mut()
                        .cpu_mut()
                        .set_gpr(0, stepper_state.local_player_index() as i32);
                })
            }),
            (self.offsets.rom.link_is_p2_ret, {
                let stepper_state = stepper_state.clone();
                Box::new(move |mut core| {
                    let stepper_state = stepper_state.lock_inner();
                    core.gba_mut()
                        .cpu_mut()
                        .set_gpr(0, stepper_state.local_player_index() as i32);
                })
            }),
            (
                self.offsets
                    .rom
                    .comm_menu_in_battle_call_comm_menu_handle_link_cable_input,
                {
                    let munger = self.munger();
                    Box::new(move |mut core| {
                        let pc = core.as_ref().gba().cpu().thumb_pc();
                        core.gba_mut().cpu_mut().set_thumb_pc(pc + 6);
                        munger.set_copy_data_input_state(core, 2);
                    })
                },
            ),
            (self.offsets.rom.round_set_ending, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_core| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_ending();
                })
            }),
            (self.offsets.rom.round_end_entry, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_core| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_ended();
                })
            }),
            (self.offsets.rom.main_read_joyflags, {
                let munger = self.munger();
                let stepper_state = stepper_state.clone();
                Box::new(move |mut core| {
                    let mut stepper_state = stepper_state.lock_inner();
                    let current_tick = stepper_state.current_tick();

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != current_tick {
                        panic!("round tick = {} but game tick = {}", current_tick, game_current_tick);
                    }

                    if current_tick == stepper_state.commit_tick() {
                        stepper_state.set_committed_state(core.save_state().expect("save committed state"));
                    }

                    let ip = match stepper_state.peek_input_pair() {
                        Some(ip) => ip.clone(),
                        None => {
                            return;
                        }
                    };

                    if ip.local.local_tick != ip.remote.local_tick {
                        stepper_state.set_anyhow_error(anyhow::anyhow!(
                            "read joyflags: local tick != remote tick (in battle tick = {}): {} != {}",
                            current_tick,
                            ip.local.local_tick,
                            ip.remote.local_tick
                        ));
                        return;
                    }

                    if ip.local.local_tick != current_tick {
                        stepper_state.set_anyhow_error(anyhow::anyhow!(
                            "read joyflags: input tick != in battle tick: {} != {}",
                            ip.local.local_tick,
                            current_tick,
                        ));
                        return;
                    }

                    core.gba_mut().cpu_mut().set_gpr(4, (ip.local.joyflags | 0xfc00) as i32);

                    if current_tick == stepper_state.dirty_tick() {
                        stepper_state.set_dirty_state(core.save_state().expect("save dirty state"));
                    }
                })
            }),
            (self.offsets.rom.copy_input_data_entry, {
                let munger = self.munger();
                let stepper_state = stepper_state.clone();
                Box::new(move |core| {
                    let mut stepper_state = stepper_state.lock_inner();
                    if stepper_state.is_round_ending() {
                        return;
                    }

                    let current_tick = stepper_state.current_tick();

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != current_tick {
                        panic!("round tick = {} but game tick = {}", current_tick, game_current_tick);
                    }

                    let ip = match stepper_state.pop_input_pair() {
                        Some(ip) => ip.clone(),
                        None => {
                            return;
                        }
                    };

                    if ip.local.local_tick != ip.remote.local_tick {
                        stepper_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: local tick != remote tick (in battle tick = {}): {} != {}",
                            current_tick,
                            ip.local.local_tick,
                            ip.remote.local_tick
                        ));
                        return;
                    }

                    if ip.local.local_tick != current_tick {
                        stepper_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: input tick != in battle tick: {} != {}",
                            ip.local.local_tick,
                            current_tick,
                        ));
                        return;
                    }

                    let local_packet = stepper_state.peek_local_packet().unwrap().clone();
                    if local_packet.tick != current_tick {
                        stepper_state.set_anyhow_error(anyhow::anyhow!(
                            "copy input data: local packet tick != in battle tick: {} != {}",
                            local_packet.tick,
                            current_tick,
                        ));
                        return;
                    }

                    munger.set_rx_packet(
                        core,
                        stepper_state.local_player_index() as u32,
                        &local_packet.packet.clone().try_into().unwrap(),
                    );
                    munger.set_rx_packet(
                        core,
                        stepper_state.remote_player_index() as u32,
                        &stepper_state
                            .apply_shadow_input(crate::input::Pair {
                                local: ip.local.with_packet(local_packet.packet),
                                remote: ip.remote,
                            })
                            .expect("apply shadow input")
                            .try_into()
                            .unwrap(),
                    );
                })
            }),
            (self.offsets.rom.copy_input_data_ret, {
                let munger = self.munger();
                let stepper_state = stepper_state.clone();
                Box::new(move |core| {
                    let mut stepper_state = stepper_state.lock_inner();
                    if stepper_state.is_round_ending() {
                        return;
                    }

                    let current_tick = stepper_state.current_tick();

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != current_tick {
                        panic!("round tick = {} but game tick = {}", current_tick, game_current_tick);
                    }

                    stepper_state.set_local_packet(current_tick + 1, munger.tx_packet(core).to_vec());
                })
            }),
            (self.offsets.rom.round_post_increment_tick, {
                let stepper_state = stepper_state.clone();
                let munger = self.munger();
                Box::new(move |core| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.increment_current_tick();
                    let current_tick = stepper_state.current_tick();

                    let game_current_tick = munger.current_tick(core);
                    if game_current_tick != current_tick {
                        stepper_state.set_anyhow_error(anyhow::anyhow!(
                            "post increment tick: round tick = {} but game tick = {}",
                            current_tick,
                            game_current_tick
                        ));
                    }
                })
            }),
            (self.offsets.rom.round_end_set_win, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_result(crate::stepper::BattleOutcome::Win);
                })
            }),
            (self.offsets.rom.round_end_set_loss, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_result(crate::stepper::BattleOutcome::Loss);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_win, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_result(crate::stepper::BattleOutcome::Win);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_loss, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_result(crate::stepper::BattleOutcome::Loss);
                })
            }),
            (self.offsets.rom.round_end_damage_judge_set_draw, {
                let stepper_state = stepper_state.clone();
                Box::new(move |_| {
                    let mut stepper_state = stepper_state.lock_inner();
                    stepper_state.set_round_result(crate::stepper::BattleOutcome::Draw);
                })
            }),
        ]
    }
}