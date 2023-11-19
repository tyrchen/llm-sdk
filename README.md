# LLM-SDK

SDK for OpenAI compatible APIs.

## Usage

Add `llm-sdk` by using `cargo add llm-sdk`.


## Features

- [x] Embedding API
- [x] Transcription & Translation API
- [x] Speech API
- [x] Chat Completion API with tools
- [ ] Chat Completion API streaming
- [ ] Chat Completion API with image input
- [x] Create Image API
- [ ] Create Image Edit API
- [ ] Create Image Variant API

As assistant API is still in Beta and is super slow, so we don't have plan to support it (and relevant file APIs) for now.

## Examples

Here are some examples of how to use the SDK:

```rust
// create image
let sdk = LlmSdk::new("https://api.openai.com/v1", "your-api-key");
let req = CreateImageRequest::new("A happy little tree");
let res = sdk.create_image(req);

// chat completion
let messages = vec![
    ChatCompletionMessage::new_system("I can answer any question you ask me.", ""),
    ChatCompletionMessage::new_user("What is human life expectancy in the world?", "user1"),
];
let req = ChatCompletionRequest::new(messages);
let res = sdk.chat_completion(req).await?;
```

For more usage, please check the test cases.
