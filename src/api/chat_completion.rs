use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};

use crate::IntoRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {}

impl IntoRequest for ChatCompletionRequest {
    fn into_request(self, client: Client) -> RequestBuilder {
        client
            .post("https://api.openai.com/v1/chat/completions")
            .json(&self)
    }
}
