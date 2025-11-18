use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<serde_json::Value>,
    pub usage: serde_json::Value,
}

#[derive(Clone)]
pub struct VllmClient {
    client: Client,
    base_url: String,
}

impl VllmClient {
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");

        Self { client, base_url }
    }

    pub async fn chat_completions(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, reqwest::Error> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json::<ChatCompletionResponse>()
            .await?;

        Ok(response)
    }

    pub async fn health_check(&self) -> Result<bool, reqwest::Error> {
        let url = format!("{}/health", self.base_url);

        let response = self.client.get(&url).send().await?;

        Ok(response.status().is_success())
    }
}

#[derive(Debug, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

impl UsageStats {
    pub fn from_json(value: &serde_json::Value) -> Self {
        Self {
            prompt_tokens: value
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            completion_tokens: value
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
            total_tokens: value
                .get("total_tokens")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32,
        }
    }
}
