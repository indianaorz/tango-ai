use fluent_templates::Loader;

use crate::i18n;

pub fn make_base_activity(
    lang: &unic_langid::LanguageIdentifier,
    game_family: Option<&str>,
) -> discord_presence::models::Activity {
    let title = game_family.map(|family| {
        i18n::LOCALES
            .lookup(lang, &format!("game-{}", family))
            .unwrap()
    });
    discord_presence::models::Activity {
        details: title.clone(),
        assets: Some(discord_presence::models::ActivityAssets {
            small_image: Some("logo".to_string()),
            small_text: Some("Tango".to_string()),
            large_image: game_family.as_ref().map(|family| family.to_string()),
            large_text: title,
        }),
        ..Default::default()
    }
}

pub fn make_looking_activity(
    link_code: &str,
    lang: &unic_langid::LanguageIdentifier,
    game_family: Option<&str>,
) -> discord_presence::models::Activity {
    discord_presence::models::Activity {
        state: Some("Looking for match".to_string()),
        secrets: Some(discord_presence::models::ActivitySecrets {
            join: Some(link_code.to_string()),
            ..Default::default()
        }),
        party: Some(discord_presence::models::ActivityParty {
            id: Some(format!("party:{}", link_code)),
            size: Some((1, 2)),
        }),
        ..make_base_activity(lang, game_family)
    }
}

pub fn make_single_player_activity(
    lang: &unic_langid::LanguageIdentifier,
    game_family: Option<&str>,
) -> discord_presence::models::Activity {
    discord_presence::models::Activity {
        state: Some("In single player".to_string()),
        ..make_base_activity(lang, game_family)
    }
}

pub fn make_in_lobby_activity(
    link_code: &str,
    lang: &unic_langid::LanguageIdentifier,
    game_family: Option<&str>,
) -> discord_presence::models::Activity {
    discord_presence::models::Activity {
        state: Some("In lobby".to_string()),
        party: Some(discord_presence::models::ActivityParty {
            id: Some(format!("party:{}", link_code)),
            size: Some((2, 2)),
        }),
        ..make_base_activity(lang, game_family)
    }
}

pub fn make_in_progress_activity(
    link_code: &str,
    start_time: std::time::SystemTime,
    lang: &unic_langid::LanguageIdentifier,
    game_family: Option<&str>,
) -> discord_presence::models::Activity {
    discord_presence::models::Activity {
        state: Some("Match in progress".to_string()),
        party: Some(discord_presence::models::ActivityParty {
            id: Some(format!("party:{}", link_code)),
            size: Some((2, 2)),
        }),
        timestamps: Some(discord_presence::models::ActivityTimestamps {
            start: start_time
                .duration_since(std::time::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_millis() as u64),
            end: None,
        }),
        ..make_base_activity(lang, game_family)
    }
}

pub struct Client {
    drpc: discord_presence::Client,
    current_activity:
        std::sync::Arc<parking_lot::Mutex<Option<discord_presence::models::Activity>>>,
    current_join_secret: std::sync::Arc<parking_lot::Mutex<Option<String>>>,
}

impl Client {
    pub fn new(client_id: u64) -> Self {
        let drpc = discord_presence::Client::new(client_id);

        let current_activity: std::sync::Arc<
            parking_lot::Mutex<Option<discord_presence::models::Activity>>,
        > = std::sync::Arc::new(parking_lot::Mutex::new(None));
        let current_join_secret = std::sync::Arc::new(parking_lot::Mutex::new(None));

        rayon::spawn({
            let mut drpc = drpc.clone();
            let current_activity = current_activity.clone();
            let current_join_secret = current_join_secret.clone();
            move || {
                drpc.start();
                drpc.on_activity_join(move |e| {
                    *current_join_secret.lock() = e
                        .event
                        .get("secret")
                        .as_ref()
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                });

                loop {
                    if let Some(activity) = current_activity.lock().as_ref() {
                        let _ = drpc.set_activity(|_| activity.clone());
                    } else {
                        let _ = drpc.clear_activity();
                    }
                    std::thread::sleep(std::time::Duration::from_secs(15));
                }
            }
        });

        let client = Self {
            drpc,
            current_activity,
            current_join_secret,
        };
        client
    }

    pub fn set_current_activity(&self, activity: Option<discord_presence::models::Activity>) {
        let mut drpc = self.drpc.clone();
        rayon::spawn({
            let mut activity = activity.clone();
            move || {
                if let Some(activity) = activity.take() {
                    let _ = drpc.set_activity(move |_| activity);
                } else {
                    let _ = drpc.clear_activity();
                }
            }
        });
        *self.current_activity.lock() = activity;
    }

    pub fn has_current_join_secret(&self) -> bool {
        self.current_join_secret.lock().is_some()
    }

    pub fn take_current_join_secret(&self) -> Option<String> {
        self.current_join_secret.lock().take()
    }
}
