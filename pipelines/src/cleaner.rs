//! 青空文庫テキストのクレンジングエンジン
//!
//! 青空文庫特有のマークアップ（ルビ記法、入力者注記、ヘッダー、底本フッターなど）を
//! 高精度に除去し、機械学習に最適なクリーンなテキストに正規化します。

/// 青空文庫テキストをクレンジングして本文のみを抽出する
pub fn clean_aozora_text(raw_text: &str) -> String {
    // 1. 本文ブロックの切り出し（ヘッダーとフッターの除去）
    let body_text = extract_body(raw_text);

    // 2. ルビ記法と入力者注記の除去
    let cleaned = strip_markup(&body_text);

    // 3. 空白行・改行の正規化
    normalize_newlines(&cleaned)
}

/// ヘッダー（タイトル・記号説明）およびフッター（底本情報）を切り離し、本文を抽出
fn extract_body(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut start_idx = 0;
    let mut end_idx = lines.len();

    // 青空文庫の区切り線 "------..." を検出
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

    // フッター（底本情報など）の開始位置を検出
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

/// ルビ記法（《...》、｜）と入力者注（［＃...］）の除去
fn strip_markup(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let ch = chars[i];

        // 1. ルビの親文字区切り記号 '｜' (全角) or '|' (半角)
        if ch == '｜' || ch == '|' {
            i += 1;
            continue;
        }

        // 2. ルビ 《...》 の除去
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

        // 3. 入力者注 ［＃...］ または [#...] の除去
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

/// 連続する空行を圧縮し、前後の余白をトリム
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
