# 実装計画書: System 1 精度改善・Gitフック組込・System 2 カスケード統合

## 1. 目的と背景
System 1 (`oniwa-decide`) における短コードスニペットの過信誤分類リスクを低減し、`git diff --cached` に基づくチャンク抽出型の高速プリコミットフック（100ms台目標、ハードブロック＆バイパス案内）を整備します。また、Raspberry Pi 4 の省メモリ環境を保護するため、System 1 の `EscalateToSystemTwo` 判定時に System 2 (`oniwa-lm`) をオンデマンド起動・推論・即時解放するカスケード統合基盤を構築します。

## 2. 非ゴールとスコープ制限
- **非ゴール**:
  - Choiceヘッドのカテゴリ体系（`DocCategory`: Rust, Python, Legal/Tech, Literature）の変更は行わない。
  - 常駐型IPCデーモン（バックグラウンド監視プロセス）の構築は行わず、オンデマンドプロセス起動・完了時即時解放で完結させる。
  - StateEmbedding（潜在中間テンソル）のバイナリ密結合伝達は避け、構造化プロンプト（コードチャンク＋判定理由・スコア）による疎結合インターフェースとする。
- **後方互換性**:
  - `DecisionConfig` や `meta.json` は `#[serde(default)]` を維持し、既存チェックポイントとの互換性を完全に保つ。

## 3. 実装詳細

### Phase B: 温度キャリブレーション ＆ 短コード回帰テスト
- **`crates/oniwa-decide/src/config.rs` & `src/model.rs`**:
  - `config.rs` を新設または分離・整備し、温度係数 $T$ の読み込み・設定メタデータ動的更新ロジックを体系化。
  - `DecisionModel::decide` および `decide_with_profile` での Choice ロジットに対する温度スケーリング除算の動作を保証。
- **短コードスニペット回帰テスト集**:
  - `crates/oniwa-decide/tests/regression_snippets.rs` を追加。
  - 短コード・構文破損・曖昧スニペットに対するキャリブレーション挙動の回帰テストを網羅。

### Phase C: 差分抽出型プリコミットフック ＆ ハードブロック整備
- **`crates/oniwa-decide/src/bin/gatekeeper.rs`**:
  - `git diff --cached` からの差分チャンク抽出を最適化。空差分・非コード差分時は安全に早期終了（exit 0）。
  - ブロック発生時の判定理由、スコア、および `--no-verify` / `ONIWA_BYPASS=1` スキップ案内を視認性高く出力。
- **`.githooks/pre-commit`**:
  - `gatekeeper --router` を組み込み、`ONIWA_BYPASS=1` による緊急脱出ハッチを実装。

### Phase A: オンデマンド System 2 カスケード統合
- **`crates/oniwa-decide/src/cascade.rs` (および `lib.rs` へのエクスポート)**:
  - `EscalateToSystemTwo` 検知時に、構造化プロンプト（差分チャンク＋確信度スコア＋エスカレーション理由）を組み立て、`oniwa-lm` プロセスをオンデマンド起動・終了するパイプライン `System2CascadeRunner` を実装。
  - Raspberry Pi 4 でのメモリ不足やプロセス起動失敗時に安全にフォールバックするエラーハンドリングを完備。
  - モック/実プロセス統合テストを追加。

## 4. 検証計画
1. `cargo test -p oniwa-decide --test regression_snippets`
2. `cargo test -p oniwa-decide`
3. `cargo test --workspace`
4. 静的解析・フォーマット:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
5. プリコミットフック動作およびバイパス確認
