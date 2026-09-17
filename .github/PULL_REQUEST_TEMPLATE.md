## 概要 (Summary)
<!-- 変更内容の要約と、解決する問題・目的を簡潔に記載してください -->

## 関連 Issue (Related Issues)
<!-- 例: Closes #123 -->
Closes #

## 実装内容 (Changes)
<!-- 主な変更点や設計上の判断を箇条書きで記載してください -->
- 

## 検証内容 (Verification)
<!-- 実施した検証手順と結果を記載してください -->
- [ ] `cargo test --workspace` が全件パスすることを確認
- [ ] `cargo fmt --all -- --check` でフォーマット違反がないことを確認
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` で警告がないことを確認
- [ ] 必要に応じて動作検証（`cargo run --release ...`）を実施

## チェックリスト (Checklist)
- [ ] [CONTRIBUTING.md](CONTRIBUTING.md) のガイドラインに準拠している
- [ ] 既存のチェックポイントやモデル設定との後方互換性が維持されている（該当する場合）
- [ ] 関連するドキュメント（README、docs/ 等）を更新した（該当する場合）
