use regex_lite::Regex;
use shlex::Shlex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tokio::fs;

static PROMPT_ARG_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\$[A-Z][A-Z0-9_]*").unwrap_or_else(|_| std::process::abort()));

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomPrompt {
    pub name: String,
    pub path: PathBuf,
    pub content: String,
    pub description: Option<String>,
    pub argument_hint: Option<String>,
}

#[derive(Debug)]
pub enum PromptArgsError {
    MissingAssignment { token: String },
    MissingKey { token: String },
}

impl PromptArgsError {
    fn describe(&self, command: &str) -> String {
        match self {
            PromptArgsError::MissingAssignment { token } => format!(
                "Could not parse {command}: expected key=value but found '{token}'. Wrap values in double quotes if they contain spaces."
            ),
            PromptArgsError::MissingKey { token } => {
                format!("Could not parse {command}: expected a name before '=' in '{token}'.")
            }
        }
    }
}

#[derive(Debug)]
pub enum PromptExpansionError {
    Args {
        command: String,
        error: PromptArgsError,
    },
    MissingArgs {
        command: String,
        missing: Vec<String>,
    },
}

impl PromptExpansionError {
    pub fn user_message(&self) -> String {
        match self {
            PromptExpansionError::Args { command, error } => error.describe(command),
            PromptExpansionError::MissingArgs { command, missing } => {
                let list = missing.join(", ");
                format!(
                    "Missing required args for {command}: {list}. Provide as key=value (quote values with spaces)."
                )
            }
        }
    }
}

/// Parse a first-line slash command of the form `/name <rest>`.
pub fn parse_slash_name(line: &str) -> Option<(&str, &str)> {
    let stripped = line.strip_prefix('/')?;
    let mut name_end = stripped.len();
    for (idx, ch) in stripped.char_indices() {
        if ch.is_whitespace() {
            name_end = idx;
            break;
        }
    }
    let name = &stripped[..name_end];
    if name.is_empty() {
        return None;
    }
    let rest = stripped[name_end..].trim_start();
    Some((name, rest))
}

fn prompt_argument_names(content: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut names = Vec::new();
    for matched in PROMPT_ARG_REGEX.find_iter(content) {
        if matched.start() > 0 && content.as_bytes()[matched.start() - 1] == b'$' {
            continue;
        }
        let name = &content[matched.start() + 1..matched.end()];
        if name == "ARGUMENTS" {
            continue;
        }
        let name = name.to_string();
        if seen.insert(name.clone()) {
            names.push(name);
        }
    }
    names
}

fn parse_prompt_inputs(rest: &str) -> Result<HashMap<String, String>, PromptArgsError> {
    let mut map = HashMap::new();
    if rest.trim().is_empty() {
        return Ok(map);
    }

    for token in Shlex::new(rest) {
        let Some((key, value)) = token.split_once('=') else {
            return Err(PromptArgsError::MissingAssignment { token });
        };
        if key.is_empty() {
            return Err(PromptArgsError::MissingKey { token });
        }
        map.insert(key.to_string(), value.to_string());
    }
    Ok(map)
}

/// Expands a saved custom prompt slash command.
///
/// Returns `Ok(None)` when the slash command does not match any saved prompt.
pub fn expand_custom_prompt(
    name: &str,
    rest: &str,
    custom_prompts: &[CustomPrompt],
) -> Result<Option<String>, PromptExpansionError> {
    let Some(prompt) = custom_prompts.iter().find(|p| p.name == name) else {
        return Ok(None);
    };

    let required = prompt_argument_names(&prompt.content);
    if !required.is_empty() {
        let inputs = parse_prompt_inputs(rest).map_err(|error| PromptExpansionError::Args {
            command: format!("/{name}"),
            error,
        })?;
        let missing = required
            .into_iter()
            .filter(|key| !inputs.contains_key(key))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(PromptExpansionError::MissingArgs {
                command: format!("/{name}"),
                missing,
            });
        }

        let content = &prompt.content;
        let replaced = PROMPT_ARG_REGEX.replace_all(content, |caps: &regex_lite::Captures<'_>| {
            if let Some(matched) = caps.get(0)
                && matched.start() > 0
                && content.as_bytes()[matched.start() - 1] == b'$'
            {
                return matched.as_str().to_string();
            }
            let whole = &caps[0];
            let key = &whole[1..];
            inputs
                .get(key)
                .cloned()
                .unwrap_or_else(|| whole.to_string())
        });
        return Ok(Some(replaced.into_owned()));
    }

    let pos_args = Shlex::new(rest).collect::<Vec<_>>();
    Ok(Some(expand_numeric_placeholders(
        &prompt.content,
        &pos_args,
    )))
}

fn expand_numeric_placeholders(content: &str, args: &[String]) -> String {
    let mut out = String::with_capacity(content.len());
    let mut i = 0;
    let mut cached_joined_args: Option<String> = None;
    while let Some(off) = content[i..].find('$') {
        let j = i + off;
        out.push_str(&content[i..j]);
        let rest = &content[j..];
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 {
            match bytes[1] {
                b'$' => {
                    out.push_str("$$");
                    i = j + 2;
                    continue;
                }
                b'1'..=b'9' => {
                    let idx = (bytes[1] - b'1') as usize;
                    if let Some(val) = args.get(idx) {
                        out.push_str(val);
                    }
                    i = j + 2;
                    continue;
                }
                _ => {}
            }
        }
        if rest.len() > "ARGUMENTS".len() && rest[1..].starts_with("ARGUMENTS") {
            if !args.is_empty() {
                let joined = cached_joined_args.get_or_insert_with(|| args.join(" "));
                out.push_str(joined);
            }
            i = j + 1 + "ARGUMENTS".len();
            continue;
        }
        out.push('$');
        i = j + 1;
    }
    out.push_str(&content[i..]);
    out
}

pub async fn discover_prompts_in(dir: &Path) -> Vec<CustomPrompt> {
    let mut prompts = Vec::new();
    let mut entries = match fs::read_dir(dir).await {
        Ok(entries) => entries,
        Err(_) => return prompts,
    };

    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let is_file = fs::metadata(&path)
            .await
            .map(|metadata| metadata.is_file())
            .unwrap_or(false);
        if !is_file {
            continue;
        }
        let is_markdown = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("md"));
        if !is_markdown {
            continue;
        }
        let Some(name) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        let content = match fs::read_to_string(&path).await {
            Ok(content) => content,
            Err(_) => continue,
        };
        let (description, argument_hint, content) = parse_frontmatter(&content);
        prompts.push(CustomPrompt {
            name,
            path,
            content,
            description,
            argument_hint,
        });
    }

    prompts.sort_by(|left, right| left.name.cmp(&right.name));
    prompts
}

fn parse_frontmatter(content: &str) -> (Option<String>, Option<String>, String) {
    let mut segments = content.split_inclusive('\n');
    let Some(first_segment) = segments.next() else {
        return (None, None, String::new());
    };
    let first_line = first_segment.trim_end_matches(['\r', '\n']);
    if first_line.trim() != "---" {
        return (None, None, content.to_string());
    }

    let mut description = None;
    let mut argument_hint = None;
    let mut frontmatter_closed = false;
    let mut consumed = first_segment.len();

    for segment in segments {
        let line = segment.trim_end_matches(['\r', '\n']);
        let trimmed = line.trim();

        if trimmed == "---" {
            frontmatter_closed = true;
            consumed += segment.len();
            break;
        }

        if trimmed.is_empty() || trimmed.starts_with('#') {
            consumed += segment.len();
            continue;
        }

        if let Some((key, value)) = trimmed.split_once(':') {
            let key = key.trim().to_ascii_lowercase();
            let mut value = value.trim().to_string();
            if value.len() >= 2 {
                let bytes = value.as_bytes();
                let first = bytes[0];
                let last = bytes[bytes.len() - 1];
                if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
                    value = value[1..value.len().saturating_sub(1)].to_string();
                }
            }
            match key.as_str() {
                "description" => description = Some(value),
                "argument-hint" | "argument_hint" => argument_hint = Some(value),
                _ => {}
            }
        }

        consumed += segment.len();
    }

    if !frontmatter_closed {
        return (None, None, content.to_string());
    }

    let body = if consumed >= content.len() {
        String::new()
    } else {
        content[consumed..].to_string()
    };
    (description, argument_hint, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn custom_prompt(name: &str, content: &str) -> CustomPrompt {
        CustomPrompt {
            name: name.to_string(),
            path: format!("/tmp/{name}.md").into(),
            content: content.to_string(),
            description: None,
            argument_hint: None,
        }
    }

    #[test]
    fn expands_named_arguments() {
        let prompts = vec![custom_prompt("custom", "Review $USER changes on $BRANCH")];
        let out = expand_custom_prompt("custom", "USER=Alice BRANCH=main", &prompts).unwrap();
        assert_eq!(out, Some("Review Alice changes on main".to_string()));
    }

    #[test]
    fn expands_positional_arguments() {
        let prompts = vec![custom_prompt(
            "custom",
            "Custom prompt with $1 and $ARGUMENTS.",
        )];
        let out = expand_custom_prompt("custom", "foo bar", &prompts).unwrap();
        assert_eq!(out, Some("Custom prompt with foo and foo bar.".to_string()));
    }

    #[test]
    fn reports_missing_named_arguments() {
        let prompts = vec![custom_prompt("custom", "Review $USER changes on $BRANCH")];
        let err = expand_custom_prompt("custom", "USER=Alice", &prompts).unwrap_err();
        assert_eq!(
            err.user_message(),
            "Missing required args for /custom: BRANCH. Provide as key=value (quote values with spaces)."
        );
    }
}
