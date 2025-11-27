# トラブルシューティングガイド - Phase 1 Week 1

## カメラ映像が表示されない問題と解決策

このドキュメントは実際に発生した問題とその解決策を記録したものです。

---

## 問題1: 映像ストリームの起動漏れ ⚠️ 最重要

### 症状
- アプリケーションは正常起動
- Web UIは表示される
- カメラは初期化される
- **しかし映像が表示されない**
- ログに `[Camera] Frame #XX` が出力されない

### 根本原因
`WebUI.process_loop()` を開始するトリガーが存在せず、UI構築後もカメラフレームの読み出し・送信が一切行われていない。

### 詳細
```python
# 問題のあるコード
class WebUI:
    def setup_ui(self):
        # UI構築
        self.setup_camera_display()
        self.setup_controls()
        self.setup_data_collection_panel()
        # ← ここで start_processing() が呼ばれていない！
    
    def start_processing(self):
        """Start processing loop."""
        if not self.running:
            asyncio.create_task(self.process_loop())
    
    async def process_loop(self):
        """Main processing loop for video streaming."""
        self.running = True
        while self.running:
            frame = self.camera.read()
            # フレーム処理・送信
            # ← このループが開始されていない！
```

### 解決策 ✅

**方法1: UI初期化後に自動起動（推奨）**
```python
def setup_ui(self):
    # UI構築
    self.setup_camera_display()
    self.setup_controls()
    self.setup_data_collection_panel()
    self.setup_status_bar()
    
    # UI初期化後0.1秒後に自動で処理開始
    ui.timer(0.1, lambda: self.start_processing(), once=True)
```

**方法2: main.pyで明示的に起動**
```python
# main.py
web_ui = WebUI(camera, model, monitor, display_config)
web_ui.setup_ui()
web_ui.start_processing()  # 明示的に起動
```

**推奨**: 方法1（UI初期化と処理開始を分離し、タイミング制御が容易）

### 検証方法
```bash
# ログで確認
tail -f jetracer_start.log

# 期待される出力:
# [WebUI] Starting process loop...
# [Camera] Frame #30: shape=(480, 640, 3), dtype=uint8
# [Camera] Frame #60: shape=(480, 640, 3), dtype=uint8
```

---

## 問題2: NiceGUI 1.4系のJavaScript実行方法の不整合

### 症状
複数のエラーが連鎖的に発生：

1. **await で二重待機**
   ```
   RuntimeError: AwaitableResponse must be awaited immediately after creation
   ```

2. **auto-indexページでのawait**
   ```
   ValueError: There are multiple clients connected. 
   It's not clear which one to wait for.
   ```

3. **respond引数エラー**
   ```
   TypeError: run_javascript() got an unexpected keyword argument 'respond'
   ```

### 根本原因
NiceGUI 1.4系でAPIが変更され、`client.run_javascript()` の使用方法が変わった：

- **旧仕様（〜1.3系）**: `await` して結果を待つ
- **新仕様（1.4系〜）**: `await` せずに呼び出す、`respond` 引数は削除

### 詳細

**問題のあるコード（NiceGUI 1.3系用）**:
```python
async def send_frame(self, frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # ❌ NiceGUI 1.4系では動かない
    await self.video_container.client.run_javascript(
        f'window.updateVideoFrame("{img_base64}")',
        respond=False  # respond引数は削除されている
    )
```

### 解決策 ✅

**NiceGUI 1.4系対応コード**:
```python
async def send_frame(self, frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # ✅ NiceGUI 1.4系: await しない、respond引数なし
    self.video_container.client.run_javascript(
        f'window.updateVideoFrame("{img_base64}")'
    )
    # 非同期実行されるが、結果を待たない（Fire and Forget方式）
```

**重要なポイント**:
1. `await` を**削除**
2. `respond` 引数を**削除**
3. JavaScriptは非同期実行される（Fire and Forget）
4. エラーハンドリングは必要に応じて追加

### バージョン互換性

| NiceGUI Version | 実装方法 |
|----------------|---------|
| 〜1.3系 | `await client.run_javascript(...)` |
| 1.4系〜 | `client.run_javascript(...)`（awaitなし） |

### 検証方法
```python
# ブラウザコンソール（F12）で確認
# 正常な場合:
# [Canvas] Update function initialized
# (エラーメッセージなし)
```

---

## 問題3: Display FPS過大による負荷

### 症状
- カメラ映像は表示される
- しかしUIが重い、反応が遅い
- CPU使用率が高い（50%以上）
- ネットワーク帯域を圧迫

### 根本原因
デフォルトのDisplay FPS（10）が高すぎる。

Jetson Orin Nanoでは：
- 640×480画像を10 FPSでエンコード＋送信 = 高負荷
- ブラウザ側のデコード処理も負荷
- 実用上5 FPSで十分

### 詳細
```python
# 問題のある設定
display:
  display_fps: 10  # 1秒間に10フレーム送信 = 高負荷
  jpeg_quality: 80
```

### 解決策 ✅

**configs/config.yaml の修正**:
```yaml
display:
  display_fps: 5   # 5 FPSに削減（50%負荷削減）
  jpeg_quality: 75 # 品質もやや下げて軽量化
  overlay_mask: false
  overlay_alpha: 0.4
```

**web_ui.py の初期値も修正**:
```python
def __init__(self, camera, model, monitor, display_config=None):
    # ...
    self.target_display_fps = display_config.get('display_fps', 5)  # 10 → 5
    self.jpeg_quality = display_config.get('jpeg_quality', 75)       # 80 → 75
```

### 負荷比較

| Display FPS | CPU使用率 | 帯域使用量 | 体感品質 |
|------------|----------|-----------|---------|
| 10 FPS | 45-60% | 3-5 Mbps | 滑らか |
| 5 FPS | 20-30% | 1.5-2.5 Mbps | 十分滑らか |
| 3 FPS | 10-15% | 1-1.5 Mbps | やや カクつく |

**推奨**: 5 FPS（品質と負荷のバランスが最適）

### 検証方法
```bash
# CPU使用率確認
top -p $(pgrep -f "python3 main.py")

# 期待値: 20-30% CPU使用率
```

---

## 補足: Canvas Script注入の正しい方法

### NiceGUI 1.4系での制約
- `ui.html()` 内に `<script>` タグを直接含められない
- エラー: `ValueError: HTML elements must not contain <script> tags`

### 正しい実装

**Step 1: Canvas HTMLのみを配置**
```python
def setup_camera_display(self):
    with ui.card().classes('w-full max-w-4xl'):
        ui.label('📹 Camera Feed').classes('text-xl mb-2')
        
        # Canvas HTMLのみ（scriptなし）
        self.video_container = ui.html('''
            <canvas id="videoCanvas" width="640" height="480" 
                    style="width: 100%; max-width: 640px; border: 2px solid #ccc;">
            </canvas>
        ''').classes('w-full')
        
        # Script を別途注入
        self._inject_canvas_script()
```

**Step 2: Script を ui.add_body_html() で注入**
```python
def _inject_canvas_script(self):
    """Inject canvas update script separately."""
    ui.add_body_html('''
        <script>
            (function() {
                // DOMContentLoaded 待機
                function initCanvas() {
                    const canvas = document.getElementById('videoCanvas');
                    if (!canvas) {
                        setTimeout(initCanvas, 100);
                        return;
                    }
                    
                    const ctx = canvas.getContext('2d');
                    
                    // グローバル関数として定義
                    window.updateVideoFrame = function(base64Data) {
                        const img = new Image();
                        img.onload = function() {
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                        };
                        img.src = 'data:image/jpeg;base64,' + base64Data;
                    };
                    
                    console.log('[Canvas] Update function initialized');
                }
                
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', initCanvas);
                } else {
                    initCanvas();
                }
            })();
        </script>
    ''')
```

**重要なポイント**:
1. Canvas要素が存在するまで待機（`setTimeout`ループ）
2. `DOMContentLoaded` イベントに対応
3. `window.updateVideoFrame` をグローバルスコープに定義
4. 初期化完了をコンソールに出力（デバッグ用）

---

## まとめ

### 修正前の問題
1. ❌ カメラ処理ループが起動しない → 映像なし
2. ❌ JavaScript実行方法が旧API → エラー連鎖
3. ❌ Display FPS高すぎ → 負荷大

### 修正後
1. ✅ `ui.timer` で自動起動 → 映像表示
2. ✅ `run_javascript` を await なしで呼び出し → 正常動作
3. ✅ Display FPS を 5 に削減 → 軽快動作

### 成功基準
- [x] カメラ映像がリアルタイム表示
- [x] CPU使用率 20-30%
- [x] UI反応性良好
- [x] データ収集機能が安定動作

---

## 参考情報

### NiceGUI バージョン確認
```bash
pip show nicegui
# Version: 1.4.33
```

### 関連ドキュメント
- NiceGUI 1.4 Migration Guide: https://nicegui.io/documentation/migration_guide
- JavaScript API Changes: https://nicegui.io/documentation/section_advanced#javascript

### トラブルシューティングコマンド
```bash
# アプリケーションログ
tail -f jetracer_start.log

# CPU/メモリ使用状況
htop

# プロセス確認
ps aux | grep python3

# ブラウザコンソール
# F12キー → Console タブ
```

---

**作成日**: 2025-11-27  
**検証環境**: Jetson Orin Nano 8GB, NiceGUI 1.4.33, Python 3.10  
**ステータス**: 検証済み・動作確認完了
