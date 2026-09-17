# 🌿 ONIWA 開発コントリビューションガイド (Contributing Guide)

<p align="left">
  <b>日本語</b> | <a href="CONTRIBUTING.en.md">English</a>
</p>

ONIWA (お庭) プロジェクトへの興味とご協力をいただき、ありがとうございます！  
本プロジェクトは、巨大データセンターや無断スクレイピングデータに頼らず、出自が100%追跡可能なクリーンなオープンデータと省電力エッジデバイスで自律育成する小さな知性（SLM）を目指しています。

バグ報告、機能提案、コードの改善、ドキュメントの加筆など、あらゆるコントリビューションを心から歓迎します。

---

## 1. コア原則（Core Principles）

コントリビューションにあたり、以下の設計思想を共有してください：

1. **ピュア Rust 原則 (Pure Rust)**:  
   外部 ML フレームワーク（PyTorch, TensorFlow, CUDA 等）への依存は一切行いません。すべての順伝播・逆伝播・オプティマイザ・データ処理をピュア Rust で完結させます。
2. **出自の完全な透明性 (Provenance & Auditability)**:  
   学習コーパス・語彙・重みのSHA-256チェックサムを監査台帳（`logs/ledger_index.jsonl`）に記録し、ブラックボックスを排除します。
3. **エッジ互換性と省電力 (Raspberry Pi 4 互換)**:  
   約 5W の省電力環境で稼働することを最優先とし、熱制御・消費電力トラッキングを阻害しない実装を徹底します。
4. **著作権法と倫理の遵守**:  
   取り込むデータは、パブリックドメイン（著作権満了）またはCC0/MIT/Apache-2.0等の明確に許諾されたオープンソース・公的オープンデータに限定されます。詳細は [データ受け入れ憲章](docs/07_data_ingestion_charter.md) を参照してください。

---

## 2. 開発環境のセットアップ

### 前提ツール
- **Rust Toolchain**: 1.75 以上（`stable` 推奨）
- **Git**

### リポジトリのクローン & Git Hook 有効化
本リポジトリにはコミット時に自動フォーマット（`cargo fmt`）と Clippy 静的解析（`cargo clippy`）を行うフックが含まれています。
```bash
git clone https://github.com/KazutoMakino/oniwa.git
cd oniwa

# コミットフックを有効化
git config core.hooksPath .githooks
```

### ビルドとテストの確認
```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

---

## 3. コントリビューション・ワークフロー

本プロジェクトでは、トレーサビリティを確保するため **Issue ドリブン開発** を推奨しています。

### ステップ 1: Issue の確認・作成
- 大きな機能追加や仕様変更を行う場合は、まず [GitHub Issues](https://github.com/KazutoMakino/oniwa/issues) で Issue を作成（または既存の Issue にコメント）し、方向性について事前に議論してください。

### ステップ 2: トピックブランチの作成
ブランチ名は以下のプレフィックスを用いたケバブケースを推奨します：
`{issue番号}/{type}/{kebab-case-description}`
- `type`: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`
- 例: `42/feat/cosine-lr-warmup-min`
```bash
git checkout -b 42/feat/cosine-lr-warmup-min
```

### ステップ 3: 実装と検証
- 変更箇所のコードを記述します。
- **全自動テストの通過を必ず確認してください**:
  ```bash
  cargo test --workspace
  ```
- コードスタイルと静的解析のチェック:
  ```bash
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings
  ```

### ステップ 4: コミットとプッシュ
Conventional Commits に準拠したコミットメッセージを推奨します：
`<type>: <簡潔な説明> (#<issue番号>)`
```bash
git add .
git commit -m "feat: 学習率コサイン減衰にウォームアップステップを追加 (#42)"
git push -u origin 42/feat/cosine-lr-warmup-min
```

### ステップ 5: プルリクエスト（PR）作成
- GitHub 上で Pull Request を作成します。
- PR の本文には概要と、対応する Issue 番号（`Closes #42` など）を明記してください。
- CI（GitHub Actions）が自動実行され、全テスト・フォーマット・Clippy チェックが検証されます。

---

## 4. AI エージェントを活用した開発プロトコル

AI コーディングアシスタント（Antigravity、Claude Code、GitHub Copilot、Gemini CLI 等）を使用して開発を進める場合は、プロジェクトルートの [**`AGENTS.md`**](AGENTS.md) に定義された自律開発プロトコルを遵守させてください。

---

## 5. コミュニティ行動規範

すべての参加者が安全かつ敬意を持って参加できるよう、[行動規範 (CODE_OF_CONDUCT.md)](CODE_OF_CONDUCT.md) を定めています。ご理解とご協力をお願いいたします。
