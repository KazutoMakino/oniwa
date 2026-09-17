//! Text cleaning engine for Aozora Bunko
//!
//! Cleans and normalizes Aozora Bunko markups (ruby annotations, editor notes,
//! headers, bibliographical footers) to produce clean text optimal for language modeling.

/// Clean Aozora Bunko text and extract main body text
pub fn clean_aozora_text(raw_text: &str) -> String {
    // 1. Extract body text block (strip header and footer)
    let body_text = extract_body(raw_text);

    // 2. Remove ruby annotations and editor notes
    let cleaned = strip_markup(&body_text);

    // 3. Normalize blank lines and newlines
    normalize_newlines(&cleaned)
}

/// Strip header (title and symbol explanation) and footer (bibliographical notes) to extract body
fn extract_body(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut start_idx = 0;
    let mut end_idx = lines.len();

    // Detect Aozora Bunko divider line "------..."
    let mut divider_count = 0;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("---") || trimmed.starts_with("───") {
            divider_count += 1;
            if divider_count == 2 {
                start_idx = i + 1;
                break;
            }
        }
    }

    // Detect start of footer notes
    for (i, line) in lines.iter().enumerate().skip(start_idx) {
        let trimmed = line.trim();
        if trimmed.starts_with("底本：") || trimmed.starts_with("底本:") {
            end_idx = i;
            break;
        }
    }

    if start_idx < end_idx {
        lines[start_idx..end_idx].join("\n")
    } else {
        text.to_string()
    }
}

/// Strip ruby annotations (《...》, ｜) and editor notes (［＃...］)
fn strip_markup(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let ch = chars[i];

        // 1. Ruby parent-word delimiter '｜' (full-width) or '|' (half-width)
        if ch == '｜' || ch == '|' {
            i += 1;
            continue;
        }

        // 2. Strip ruby 《...》
        if ch == '《' {
            i += 1;
            while i < n && chars[i] != '》' {
                i += 1;
            }
            if i < n && chars[i] == '》' {
                i += 1;
            }
            continue;
        }

        // 3. Strip editor notes ［＃...］ or [#...]
        if ch == '［' && i + 1 < n && chars[i + 1] == '＃' {
            i += 2;
            while i < n && chars[i] != '］' {
                i += 1;
            }
            if i < n && chars[i] == '］' {
                i += 1;
            }
            continue;
        }
        if ch == '[' && i + 1 < n && chars[i + 1] == '#' {
            i += 2;
            while i < n && chars[i] != ']' {
                i += 1;
            }
            if i < n && chars[i] == ']' {
                i += 1;
            }
            continue;
        }

        result.push(ch);
        i += 1;
    }

    result
}

/// Compress consecutive blank lines and trim whitespace
fn normalize_newlines(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut empty_line_count = 0;

    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            empty_line_count += 1;
            if empty_line_count <= 1 {
                normalized.push('\n');
            }
        } else {
            empty_line_count = 0;
            normalized.push_str(trimmed);
            normalized.push('\n');
        }
    }

    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ruby_removal() {
        let input = "その時、｜李徴《りちょう》は臆病《おくびょう》な自尊心を飼っていた。";
        let cleaned = strip_markup(input);
        assert_eq!(cleaned, "その時、李徴は臆病な自尊心を飼っていた。");
    }

    #[test]
    fn test_note_removal() {
        let input = "メロスは激怒した。［＃３字下げ］必ず、かの邪智暴虐の王を除かなければならぬと決意した。";
        let cleaned = strip_markup(input);
        assert_eq!(
            cleaned,
            "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。"
        );
    }

    #[test]
    fn test_full_aozora_clean() {
        let raw = r#"山月記
中島敦

-------------------------------------------------------
【テキスト中に現れる記号について】
《》：ルビ
-------------------------------------------------------

隴西《ろうせい》の｜李徴《りちょう》は儁才《しゅんさい》にして若くして名を登科に連ね、［＃改ページ］
天宝の末年に至りて、遂に虎と化せり。

底本：「中島敦全集1」ちくま文庫、筑摩書房
1993（平成5）年1月21日第1刷発行
"#;
        let cleaned = clean_aozora_text(raw);
        assert!(cleaned.contains("隴西の李徴は儁才にして若くして名を登科に連ね、"));
        assert!(cleaned.contains("天宝の末年に至りて、遂に虎と化せり。"));
        assert!(!cleaned.contains("底本："));
        assert!(!cleaned.contains("テキスト中に現れる記号について"));
        assert!(!cleaned.contains("《"));
        assert!(!cleaned.contains("［＃"));
    }
}
