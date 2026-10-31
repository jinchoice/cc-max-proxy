use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::error::ProxyError;
use crate::types::{CliMessage, ContentBlock, MessageContent, MessagesRequest};

/// Resolve the claude CLI binary path.
pub fn resolve_claude_path() -> Result<String, ProxyError> {
    if let Ok(path) = std::env::var("CLAUDE_PATH") {
        if std::path::Path::new(&path).exists() {
            return Ok(path);
        }
        return Err(ProxyError::CliNotFound(path));
    }

    // Try common locations
    for candidate in &["/opt/homebrew/bin/claude", "/usr/local/bin/claude"] {
        if std::path::Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }

    // Try PATH via `which`
    if let Ok(output) = std::process::Command::new("which").arg("claude").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    Err(ProxyError::CliNotFound("claude".to_string()))
}

/// Map an Anthropic model string to a claude CLI model flag.
pub(crate) fn map_model(model: &str) -> &str {
    if model.contains("opus") {
        "opus"
    } else if model.contains("haiku") {
        "haiku"
    } else {
        "sonnet"
    }
}

/// Build the content of the single user turn sent to the CLI.
///
/// The conversation is flattened into a "Human: … / Assistant: …" text
/// transcript, but image blocks stay where they were so the CLI passes them
/// on to the model.
pub(crate) fn build_prompt(request: &MessagesRequest) -> Vec<ContentBlock> {
    let mut prompt = Vec::new();

    // System prompt
    if let Some(ref system) = request.system {
        match system {
            serde_json::Value::String(s) => push_text(&mut prompt, "\n\n", s),
            serde_json::Value::Array(blocks) => {
                let text: String = blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    push_text(&mut prompt, "\n\n", &text);
                }
            }
            _ => {}
        }
    }

    // Conversation messages
    for msg in &request.messages {
        let role = match msg.role.as_str() {
            "assistant" => "Assistant",
            _ => "Human",
        };
        let content: Vec<ContentBlock> = match &msg.content {
            MessageContent::Text(s) => vec![ContentBlock::Text { text: s.clone() }],
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| {
                    let text = match b {
                        ContentBlock::Image { .. } => return Some(b.clone()),
                        ContentBlock::Text { text } => text.clone(),
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                        } => format!("[tool_result for {}]: {}", tool_use_id, content),
                        ContentBlock::ToolUse { id, name, input } => {
                            format!("[tool_use {} ({}): {}]", name, id, input)
                        }
                        ContentBlock::Thinking { thinking } if thinking.is_empty() => return None,
                        ContentBlock::Thinking { thinking } => {
                            format!("<thinking>{}</thinking>", thinking)
                        }
                    };
                    Some(ContentBlock::Text { text })
                })
                .collect(),
        };
        // Skip messages that would produce empty content
        if content
            .iter()
            .all(|b| matches!(b, ContentBlock::Text { text } if text.trim().is_empty()))
        {
            continue;
        }
        push_text(&mut prompt, "\n\n", &format!("{}: ", role));
        let mut sep = "";
        for block in content {
            match block {
                ContentBlock::Text { text } => push_text(&mut prompt, sep, &text),
                image => prompt.push(image),
            }
            sep = "\n";
        }
    }

    prompt
}

/// Append `text` to the trailing text block, joined by `sep`, or start a new
/// text block if the prompt is empty or ends with an image.
fn push_text(prompt: &mut Vec<ContentBlock>, sep: &str, text: &str) {
    match prompt.last_mut() {
        Some(ContentBlock::Text { text: last }) => {
            last.push_str(sep);
            last.push_str(text);
        }
        _ => prompt.push(ContentBlock::Text {
            text: text.to_string(),
        }),
    }
}

/// Build the CLI's stdin: one `--input-format stream-json` user message.
pub(crate) fn build_stdin(request: &MessagesRequest) -> String {
    let message = serde_json::json!({
        "type": "user",
        "message": {"role": "user", "content": build_prompt(request)},
    });
    format!("{message}\n")
}

/// Spawn the claude CLI and stream NDJSON messages back via a channel.
///
/// `working_dir` sets the CLI's cwd (client's project dir). Falls back to `~/`.
pub async fn spawn_stream(
    claude_path: &str,
    request: &MessagesRequest,
    working_dir: Option<&str>,
) -> Result<mpsc::Receiver<Result<CliMessage, ProxyError>>, ProxyError> {
    let input = build_stdin(request);
    let model = map_model(&request.model);

    let cwd = working_dir
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("/")));

    tracing::info!(
        "Spawning claude CLI: model={}, input_len={}, cwd={}",
        model,
        input.len(),
        cwd.display()
    );

    let mut child = Command::new(claude_path)
        .env_remove("CLAUDECODE")
        .env_remove("CLAUDE_CODE_ENTRYPOINT")
        .arg("-p")
        .arg("--input-format")
        .arg("stream-json")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--verbose")
        .arg("--include-partial-messages")
        .arg("--no-session-persistence")
        .arg("--dangerously-skip-permissions")
        .arg("--permission-mode")
        .arg("bypassPermissions")
        .arg("--add-dir")
        .arg("/")
        .arg("--model")
        .arg(model)
        .current_dir(&cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Write prompt via stdin to avoid leaking it in `ps aux`
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ProxyError::CliError("failed to capture stdin".to_string()))?;
    let input_bytes = input.into_bytes();
    tokio::spawn(async move {
        let _ = stdin.write_all(&input_bytes).await;
        let _ = stdin.shutdown().await;
    });

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ProxyError::CliError("failed to capture stdout".to_string()))?;

    let (tx, rx) = mpsc::channel::<Result<CliMessage, ProxyError>>(32);

    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            match serde_json::from_str::<CliMessage>(&line) {
                Ok(msg) => {
                    tracing::debug!("CLI → {:?}", std::mem::discriminant(&msg));
                    if tx.send(Ok(msg)).await.is_err() {
                        break; // receiver dropped
                    }
                }
                Err(e) => {
                    tracing::debug!(
                        "Skipping unparseable CLI line: {} — {}",
                        e,
                        &line[..line.len().min(200)]
                    );
                }
            }
        }

        // Wait for process to exit
        match child.wait().await {
            Ok(status) if !status.success() => {
                tracing::warn!("claude CLI exited with status: {}", status);
            }
            Err(e) => {
                tracing::error!("Failed to wait on claude CLI: {}", e);
            }
            _ => {}
        }
    });

    Ok(rx)
}
