use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub database_url: String,
    pub redis_url: String,
    pub host: String,
    pub port: u16,
    pub log_level: String,
}

impl Config {
    pub fn from_env() -> Result<Self, config::ConfigError> {
        dotenvy::dotenv().ok();

        config::Config::builder()
            .add_source(config::Environment::default().separator("__"))
            .set_default("host", "0.0.0.0")?
            .set_default("port", 8000)?
            .set_default("log_level", "info")?
            .set_default("database_url", "postgresql://postgres:password@localhost:5432/distributed_llm")?
            .set_default("redis_url", "redis://localhost:6379")?
            .build()?
            .try_deserialize()
    }
}
