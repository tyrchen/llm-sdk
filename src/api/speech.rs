use crate::IntoRequest;
use derive_builder::Builder;
use reqwest::{Client, RequestBuilder};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Builder)]
#[builder(pattern = "mutable")]
pub struct SpeechRequest {
    /// One of the available TTS models: tts-1 or tts-1-hd
    #[builder(default)]
    model: SpeechModel,
    /// The text to generate audio for. The maximum length is 4096 characters.
    #[builder(setter(into))]
    input: String,
    /// The voice to use when generating the audio. Supported voices are alloy, echo, fable, onyx, nova, and shimmer. Previews of the voices are available in the Text to speech guide.
    #[builder(default)]
    voice: SpeechVoice,
    /// The format to audio in. Supported formats are mp3, opus, aac, and flac.
    #[builder(default)]
    response_format: SpeechResponseFormat,
    /// The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default.
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum SpeechModel {
    #[default]
    #[serde(rename = "tts-1")]
    Tts1,
    #[serde(rename = "tts-1-hd")]
    Tts1Hd,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechVoice {
    Alloy,
    Echo,
    Fable,
    Onyx,
    #[default]
    Nova,
    Shimmer,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechResponseFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
}

impl IntoRequest for SpeechRequest {
    fn into_request(self, client: Client) -> RequestBuilder {
        client
            .post("https://api.openai.com/v1/audio/speech")
            .json(&self)
    }
}

impl SpeechRequest {
    pub fn new(input: impl Into<String>) -> Self {
        SpeechRequestBuilder::default()
            .input(input)
            .build()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::LlmSdk;

    use super::*;
    use anyhow::Result;

    #[tokio::test]
    async fn speech_should_work() -> Result<()> {
        let sdk = LlmSdk::new(std::env::var("OPENAI_API_KEY")?);
        let req = SpeechRequest::new("The quick brown fox jumped over the lazy dog.");
        let res = sdk.speech(req).await?;

        fs::write("/tmp/llm-sdk/test.mp3", res)?;

        Ok(())
    }
}
