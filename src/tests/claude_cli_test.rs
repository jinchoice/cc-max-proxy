use serde_json::json;

use crate::claude_cli::{build_prompt, build_stdin, map_model};
use crate::types::{ContentBlock, Message, MessageContent, MessagesRequest};

/// The prompt's text, asserting it is a single text block.
fn prompt_text(req: &MessagesRequest) -> String {
    match build_prompt(req).as_slice() {
        [ContentBlock::Text { text }] => text.clone(),
        other => panic!("expected one text block, got {other:?}"),
    }
}

fn image(data: &str) -> ContentBlock {
    ContentBlock::Image {
        source: json!({"type": "base64", "media_type": "image/jpeg", "data": data}),
    }
}

fn image_json(data: &str) -> serde_json::Value {
    json!({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
}

fn text(s: &str) -> ContentBlock {
    ContentBlock::Text {
        text: s.to_string(),
    }
}

fn message(role: &str, blocks: Vec<ContentBlock>) -> Message {
    Message {
        role: role.to_string(),
        content: MessageContent::Blocks(blocks),
    }
}

fn request(messages: Vec<Message>) -> MessagesRequest {
    MessagesRequest {
        model: "claude-opus-4-6".to_string(),
        messages,
        system: None,
        stream: true,
    }
}

#[test]
fn map_model_opus() {
    assert_eq!(map_model("claude-opus-4-20250514"), "opus");
    assert_eq!(map_model("claude-opus-4-6"), "opus");
}

#[test]
fn map_model_haiku() {
    assert_eq!(map_model("claude-haiku-4-5-20251001"), "haiku");
    assert_eq!(map_model("claude-3-haiku"), "haiku");
}

#[test]
fn map_model_sonnet() {
    assert_eq!(map_model("claude-sonnet-4-20250514"), "sonnet");
    assert_eq!(map_model("claude-sonnet-4-6"), "sonnet");
}

#[test]
fn map_model_defaults_to_sonnet() {
    assert_eq!(map_model("claude-4"), "sonnet");
    assert_eq!(map_model("some-unknown-model"), "sonnet");
}

#[test]
fn build_prompt_simple_message() {
    let req = MessagesRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hello!".to_string()),
        }],
        system: None,
        stream: true,
    };

    let prompt = prompt_text(&req);
    assert_eq!(prompt, "Human: Hello!");
}

#[test]
fn build_prompt_with_system_string() {
    let req = MessagesRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hi".to_string()),
        }],
        system: Some(serde_json::Value::String("Be helpful.".to_string())),
        stream: true,
    };

    let prompt = prompt_text(&req);
    assert!(prompt.starts_with("Be helpful."));
    assert!(prompt.contains("Human: Hi"));
}

#[test]
fn build_prompt_with_system_blocks() {
    let req = MessagesRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("Hi".to_string()),
        }],
        system: Some(serde_json::json!([
            {"type": "text", "text": "Block one"},
            {"type": "text", "text": "Block two"}
        ])),
        stream: true,
    };

    let prompt = prompt_text(&req);
    assert!(prompt.starts_with("Block one\nBlock two"));
}

#[test]
fn build_prompt_multi_turn() {
    let req = MessagesRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Text("Hi there!".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("How are you?".to_string()),
            },
        ],
        system: None,
        stream: true,
    };

    let prompt = prompt_text(&req);
    assert!(prompt.contains("Human: Hello"));
    assert!(prompt.contains("Assistant: Hi there!"));
    assert!(prompt.contains("Human: How are you?"));
}

#[test]
fn build_prompt_keeps_images_in_order() {
    let req = request(vec![message(
        "user",
        vec![image("A"), image("B"), text("Which numbers?")],
    )]);

    let content = serde_json::to_value(build_prompt(&req)).unwrap();
    assert_eq!(
        content,
        json!([
            {"type": "text", "text": "Human: "},
            image_json("A"),
            image_json("B"),
            {"type": "text", "text": "Which numbers?"},
        ])
    );
}

#[test]
fn build_prompt_keeps_images_in_place_across_turns() {
    let req = request(vec![
        message("user", vec![image("A"), text("first")]),
        message("assistant", vec![text("ok")]),
        message("user", vec![text("and"), image("B")]),
    ]);

    let content = serde_json::to_value(build_prompt(&req)).unwrap();
    assert_eq!(
        content,
        json!([
            {"type": "text", "text": "Human: "},
            image_json("A"),
            {"type": "text", "text": "first\n\nAssistant: ok\n\nHuman: and"},
            image_json("B"),
        ])
    );
}

#[test]
fn build_prompt_keeps_image_only_message() {
    let req = request(vec![message("user", vec![image("A")])]);

    let content = serde_json::to_value(build_prompt(&req)).unwrap();
    assert_eq!(
        content,
        json!([{"type": "text", "text": "Human: "}, image_json("A")])
    );
}

#[test]
fn build_stdin_is_one_stream_json_user_message() {
    let req = request(vec![message("user", vec![image("A"), text("Hi")])]);

    let line = build_stdin(&req);
    assert!(line.ends_with('\n'));
    assert_eq!(line.matches('\n').count(), 1);

    let msg: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(msg["type"], "user");
    assert_eq!(msg["message"]["role"], "user");
    assert_eq!(
        msg["message"]["content"],
        serde_json::to_value(build_prompt(&req)).unwrap()
    );
}

#[test]
fn cli_path_env_override() {
    // Set CLAUDE_PATH to a non-existent path — should return error
    // SAFETY: single-threaded test, no other threads reading this env var
    unsafe {
        std::env::set_var("CLAUDE_PATH", "/tmp/nonexistent-claude-binary");
    }
    let result = crate::claude_cli::resolve_claude_path();
    unsafe {
        std::env::remove_var("CLAUDE_PATH");
    }
    assert!(result.is_err());
}
